package com.age.myapplication;


import android.annotation.SuppressLint;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import org.pytorch.IValue;
import org.pytorch.MemoryFormat;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class MainActivity extends AppCompatActivity {
    Module module;
    ImageView imageView;
    Button take_photo, test_default_image;
    TextView textView;
    int imageSize = 128;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imageView);
        textView = findViewById(R.id.tvResults);
        take_photo = findViewById(R.id.btnCapture);
        test_default_image = findViewById(R.id.btnTest);

        Bitmap defaultBitmap = BitmapFactory.decodeResource(getResources(), R.drawable.testimage);
        imageView.setImageBitmap(defaultBitmap);

        try {
            module = Module.load(assetFilePath(this, "model.ptl"));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        take_photo.setOnClickListener(view -> {
            if (checkSelfPermission(android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(cameraIntent, 3);
            } else {
                requestPermissions(new String[]{android.Manifest.permission.CAMERA}, 100);
            }
        });

        test_default_image.setOnClickListener(view -> {
            Bitmap resizedBitmap = Bitmap.createScaledBitmap(defaultBitmap, imageSize, imageSize, true);
            imageView.setImageBitmap(resizedBitmap);
            classifyImage(resizedBitmap, module);
        });
    }

    @SuppressLint("SetTextI18n")
    public void classifyImage(Bitmap image, Module module) {
        if (module == null) {
            textView.setText("Model not loaded!!!");
            return;
        }

        Bitmap resizedImage = Bitmap.createScaledBitmap(image, imageSize, imageSize, true);
        Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                resizedImage,
                new float[]{0.485f, 0.456f, 0.406f},
                new float[]{0.229f, 0.224f, 0.225f},
                MemoryFormat.CHANNELS_LAST
        );

        IValue output = module.forward(IValue.from(inputTensor));
        IValue[] outputTuple = output.toTuple();

        decodeModelOutput(outputTuple);
    }

    @SuppressLint("SetTextI18n")
    private void decodeModelOutput(IValue[] outputTuple) {
        String[] ageLabels = {"Child (0-12)", "Teen (13-19)", "Young Adult (20-28)",
                "Adult (29-55)", "Elderly (56+)"};
        String[] emotionLabels = {"Angry", "Happy", "Neutral", "Sad", "Surprised"};

        Tensor ageTensor = outputTuple[0].toTensor();
        Tensor genderTensor = outputTuple[1].toTensor();
        Tensor emotionTensor = outputTuple[2].toTensor();

        float[] ageScores = ageTensor.getDataAsFloatArray();
        float[] genderScores = genderTensor.getDataAsFloatArray();
        float[] emotionScores = emotionTensor.getDataAsFloatArray();

        int ageIndex = getMaxIndex(ageScores);
        int genderIndex = getMaxIndex(genderScores);
        int emotionIndex = getMaxIndex(emotionScores);

        String age = ageLabels[ageIndex];
        String gender = (genderIndex == 0) ? "Male" : "Female";
        String emotion = emotionLabels[emotionIndex];

        textView.setText("Predicted Age: " + age + "\n" +
                "Predicted Gender: " + gender + "\n" +
                "Predicted Emotion: " + emotion);
    }

    private int getMaxIndex(float[] array) {
        float maxScore = -Float.MAX_VALUE;
        int maxIndex = -1;
        for (int i = 0; i < array.length; i++) {
            if (array[i] > maxScore) {
                maxScore = array[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) return file.getAbsolutePath();

        try (InputStream is = context.getAssets().open(assetName);
             OutputStream os = new FileOutputStream(file)) {
            byte[] buffer = new byte[4 * 1024];
            int read;
            while ((read = is.read(buffer)) != -1) os.write(buffer, 0, read);
            os.flush();
        }
        return file.getAbsolutePath();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && requestCode == 3) {
            Bitmap image = (Bitmap) data.getExtras().get("data");
            imageView.setImageBitmap(image);
            Bitmap scaledImage = Bitmap.createScaledBitmap(image, imageSize, imageSize, true);
            classifyImage(scaledImage, module);
        }
    }
}
