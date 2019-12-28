/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.LinkedList;
import java.util.List;
import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;
import android.os.Bundle;
import android.os.Bundle;
import android.Manifest;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;
import android.renderscript.ScriptGroup;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  // Configuration values for the prepackaged SSD model.
  private static final int TF_OD_API_INPUT_SIZE = 300;
  private static final boolean TF_OD_API_IS_QUANTIZED = true;
  private static final String TF_OD_API_MODEL_FILE = "detect.tflite";
  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";
  private static final DetectorMode MODE = DetectorMode.TF_OD_API;
  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
  private static final boolean MAINTAIN_ASPECT = false;
  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;
  OverlayView trackingOverlay;
  private Integer sensorOrientation;

  private Classifier detector;

  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;
  private Bitmap mSourceBitmap = null;

  private boolean computingDetection = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  private BorderedText borderedText;



  //保存するファイル名
  private String fileName;
  private String fileNamebase = "face_picture_";
  //intをstringにするときは, String.valueoOf(int)を利用する
  private String fileType = ".jpg";
  private int fileNameCounter = 0;
  //  File path = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
  private int REQUEST_PERMISSION = 1000;


  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);

    int cropSize = TF_OD_API_INPUT_SIZE;

    try {
      detector =
          TFLiteObjectDetectionAPIModel.create(
              getAssets(),
              TF_OD_API_MODEL_FILE,
              TF_OD_API_LABELS_FILE,
              TF_OD_API_INPUT_SIZE,
              TF_OD_API_IS_QUANTIZED);
      cropSize = TF_OD_API_INPUT_SIZE;
    } catch (final IOException e) {
      e.printStackTrace();
      LOGGER.e(e, "Exception initializing classifier!");
      Toast toast =
          Toast.makeText(
              getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            cropSize, cropSize,
            sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    trackingOverlay.addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            tracker.draw(canvas);
            if (isDebug()) {
              tracker.drawDebug(canvas);
            }
          }
        });

    tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
  }

  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;
    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            LOGGER.i("Running detection on image " + currTimestamp);
            final long startTime = SystemClock.uptimeMillis();
            final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
            System.out.println("%%%%%%%%%%%" + results);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            mSourceBitmap = Bitmap.createBitmap(cropCopyBitmap);


            final Canvas canvas = new Canvas(cropCopyBitmap);
            final Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Style.STROKE);
            paint.setStrokeWidth(2.0f);

            float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
            switch (MODE) {
              case TF_OD_API:
                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                break;
            }

            final List<Classifier.Recognition> mappedRecognitions =
                new LinkedList<Classifier.Recognition>();

            for (final Classifier.Recognition result : results) {
              float left = result.getLocation().left;
              float top = result.getLocation().top;
              float right = result.getLocation().right;
              float bottom = result.getLocation().bottom;

              final RectF location = result.getLocation();
              if (location != null && result.getConfidence() >= minimumConfidence) {
                canvas.drawRect(location, paint);

                cropToFrameTransform.mapRect(location);

                result.setLocation(location);
                mappedRecognitions.add(result);
                //このへんでpresonなら切り取って出力みたいなことをしたい
                //結果のtextはどこに入っているんだ?
                System.out.println("&&&&&&&"+result.getTitle());
                //stringの比較は"=="じゃだめ, あとで直す
               //if(result.getTitle() == "bottle"){
                 if(result.getTitle().equals("person")){
                   float SMALLER = 0.950f;
                   System.out.println("SMARER>>" + SMALLER);
//                   float left = location.left;
//                   float top = result.getLocation().top;
//                   float right = result.getLocation().right;
//                   float bottom = result.getLocation().bottom;

                   System.out.println(left);
                   System.out.println(top);
                   System.out.println(right);
                   System.out.println(bottom);
                   float CUNNUM = 290f;
                   if(left > 1 && left < CUNNUM && top > 1 && top < CUNNUM && right < CUNNUM && bottom <CUNNUM) {

                     System.out.println(left);
                     System.out.println(top);
                     System.out.println(right);
                     System.out.println(bottom);
                     left = left * SMALLER;
                     top = top * SMALLER;
                     System.out.println("SMARAL" + left);
                     System.out.println("SMARAL" + top);
                     if (left < 1.0) {
                       left = 1;
                     }
                     if (top < 1.0) {
                       top = 1;
                     }
                     if (right > 300.0) {
                       right = 299;
                     }
                     if (bottom > 300.0) {
                       bottom = 299;
                     }

                     System.out.println(left);
                     System.out.println(top);
                     System.out.println(right);
                     System.out.println(bottom);


                     right = right * SMALLER;
                     bottom = bottom * SMALLER;

                     System.out.println(left);
                     System.out.println(top);
                     System.out.println(right);
                     System.out.println(bottom);
                     //トリミングコード
                     int nWidth = (int) (right - left);
                     int nHeight = (int) (bottom - top);
                     int startX = (int) (left + 15);
                     int startY = (int) top + 15;

                     System.out.println("nWidth:" + nWidth);
                     System.out.println(nHeight);
                     System.out.println(startX);
                     System.out.println(startY);
                     System.out.println("##before");
                     //mSourceBitmap = croppedBitmap.createBitmap(mSourceBitmap, nWidth, nHeight, startX, startY,null,true);
                     mSourceBitmap = croppedBitmap.createBitmap(mSourceBitmap, startX, startY, nWidth,nHeight, null, true);
                     System.out.println("##after");

                     //保存するためのコード
                     fileNameCounter++;
                     File path = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
                     fileName = fileNamebase + String.valueOf(fileNameCounter) + fileType;
                     File file = new File(path, fileName);

                     //Android23以上のときはパーミション確認
                     if (Build.VERSION.SDK_INT >= 23) {
                       checkPermission1();
                     }
                     saveFileMethod(file, mSourceBitmap);

                     //ここまで保存するコード
                   }
               }
              }
            }

            tracker.trackResults(mappedRecognitions, currTimestamp);
            trackingOverlay.postInvalidate();

            computingDetection = false;

            runOnUiThread(
                new Runnable() {
                  @Override
                  public void run() {
                    showFrameInfo(previewWidth + "x" + previewHeight);
                    showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                    showInference(lastProcessingTimeMs + "ms");
                  }
                });
          }
        });
  }

  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.
  private enum DetectorMode {
    TF_OD_API;
  }

  @Override
  protected void setUseNNAPI(final boolean isChecked) {
    runInBackground(() -> detector.setUseNNAPI(isChecked));
  }

  @Override
  protected void setNumThreads(final int numThreads) {
    runInBackground(() -> detector.setNumThreads(numThreads));
  }

  // permissionの確認
  public void checkPermission1() {
    // 既に許可している
    if (ActivityCompat.checkSelfPermission(this,
            Manifest.permission.WRITE_EXTERNAL_STORAGE)
            == PackageManager.PERMISSION_GRANTED){
      //setUpWriteExternalStorage();
    }
    // 拒否していた場合
    else{
      requestLocationPermission();
    }
  }

  // 許可を求める
  private void requestLocationPermission() {
    if (ActivityCompat.shouldShowRequestPermissionRationale(this,
            Manifest.permission.WRITE_EXTERNAL_STORAGE)) {
      //もしかしたら, Savefileの部分が違っているかも
      ActivityCompat.requestPermissions(DetectorActivity.this,
              new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
              REQUEST_PERMISSION);
    } else {
      Toast toast = Toast.makeText(this, "許可してください", Toast.LENGTH_SHORT);
      toast.show();

      ActivityCompat.requestPermissions(this,
              new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE,},
              REQUEST_PERMISSION);
    }
  }

  // 結果の受け取り
  @Override
  public void onRequestPermissionsResult(
          int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
    if (requestCode == REQUEST_PERMISSION) {
      // 使用が許可された
      if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
        //setUpWriteExternalStorage();
      } else {
        // それでも拒否された時の対応
        Toast toast = Toast.makeText(this, "何もできません", Toast.LENGTH_SHORT);
        toast.show();
      }
    }
  }
  // アンドロイドのデータベースへ登録する
  private void registerDatabase(File file) {
    ContentValues contentValues = new ContentValues();
    ContentResolver contentResolver = DetectorActivity.this.getContentResolver();
    contentValues.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");
    contentValues.put("_data", file.getAbsolutePath());
    contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            contentValues);
  }

  public void saveFileMethod(File file, Bitmap bitmap) {
    if (bitmap != null) {
      try {
        FileOutputStream output = new FileOutputStream(file);
        registerDatabase(file);
        bitmap.compress(Bitmap.CompressFormat.JPEG,100,output);
        System.out.println("OKOKOKOKOKOKO");
        System.out.println(fileName);
      } catch (IOException e) {
        e.printStackTrace();
      }
    }
    else{
      System.out.println("#####: bitmapがnullです");
    }
  }



}
