package picodiploma.dicoding.tamam_app

import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import picodiploma.dicoding.tamam_app.ml.Tamam
import java.nio.ByteBuffer

class MainActivity : AppCompatActivity() {

//    bitmap untuk gambar
    lateinit var bitmap :Bitmap
    lateinit var imgview: ImageView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imgview = findViewById(R.id.imageView)

//        read file labels.txt
        val fileName = "labels.txt"
        val inputString = application.assets.open(fileName).bufferedReader().use { it.readText() }
        var townList = inputString.split("\n")

        var tv: TextView = findViewById(R.id.textView)

        var select: Button = findViewById(R.id.button)

        select.setOnClickListener(View.OnClickListener {
//            buat intent
            var intent: Intent = Intent(Intent.ACTION_GET_CONTENT)

//            SHOW INTENT
            intent.type = "image/*"

//            start the activity of intent our make it
            startActivityForResult(intent, 100)
        })

        var predict: Button = findViewById(R.id.button2)
        predict.setOnClickListener(View.OnClickListener {
//            RESIZE BITMAP
            var resized: Bitmap = Bitmap.createScaledBitmap(bitmap, 28,28, false)
//            MODEL
            val model = Tamam.newInstance(this)

// Creates inputs for reference.
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 28, 28, 3),  DataType.FLOAT32)
//            buat byteBuffer
            var tbuffer = TensorImage.fromBitmap(resized)
//            var byteBuffer = tbuffer.buffer
            val byteBuffer = ByteBuffer.allocateDirect(28*28*3*4)
            byteBuffer.put(tbuffer.buffer)
            inputFeature0.loadBuffer(byteBuffer)

//

// Runs model inference and gets result.
            val outputs = model.process(inputFeature0)
//            output dr prediksi
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer
// get max value with func our make it below
            var max= getMax(outputFeature0.floatArray)
//            set textview
//            tv.setText(outputFeature0.floatArray[1].toString())
            tv.setText(townList[max])
// Releases model resources if no longer used.
            model.close()
        })
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        imgview.setImageURI(data?.data)

        var uri: Uri?= data?.data

        bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)


    }

//    function get max value
    fun getMax(arr: FloatArray) : Int{
        var ind=0
        var min = 0.0f

        for(i in 0..8)
        {
            if(arr[i]>min)
            {
                ind = i
                min = arr[i]
            }
        }
        return ind
    }
}