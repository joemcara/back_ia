from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf 
from tensorflow.keras import layers, initializers
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io, os

class AttentionLayer(layers.Layer):
    def build(self, input_shape):
        ch = input_shape[-1]
        # 1) concatenamos q‖k
        self.qk_conv = layers.Conv2D(
            ch * 2, 1, use_bias=False,
            kernel_initializer=initializers.HeUniform(),
            name=f'{self.name}_qk_conv'
        )
        self.qk_bn = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5,
            name=f'{self.name}_qk_bn'
        )
        # 2) proyección final
        self.proj_conv = layers.Conv2D(
            ch, 1, use_bias=False,
            kernel_initializer=initializers.HeUniform(),
            name=f'{self.name}_proj_conv'
        )
        self.proj_bn = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5,
            name=f'{self.name}_proj_bn'
        )
        # 3) positional encoding (depthwise 3×3)
        self.pe_conv = layers.Conv2D(
            ch, 3, 1, padding="same", groups=ch, use_bias=False,
            kernel_initializer=initializers.HeUniform(),
            name=f'{self.name}_pe_conv'
        )
        self.pe_bn = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5,
            name=f'{self.name}_pe_bn'
        )
        super().build(input_shape)

    def call(self, x):
        # --- self-attention ---
        qk = self.qk_bn(self.qk_conv(x))      # (B, H, W, 2C)
        q, k = tf.split(qk, 2, axis=-1)       # (B, H, W, C) ×2

        # re-shape a (B, HW, C)
        v = tf.reshape(x, (tf.shape(x)[0], -1, tf.shape(x)[-1]))
        q = tf.reshape(q, (tf.shape(x)[0], -1, tf.shape(x)[-1]))
        k = tf.reshape(k, (tf.shape(x)[0], -1, tf.shape(x)[-1]))

        # atención escalada
        attn = tf.nn.softmax(
            tf.matmul(q, k, transpose_b=True) /
            tf.math.sqrt(tf.cast(tf.shape(x)[-1], tf.float32)),
            axis=-1
        )
        out = tf.matmul(attn, v)              # (B, HW, C)
        out = tf.reshape(out, tf.shape(x))    # (B, H, W, C)

        # proyección + positional encoding + residual
        out = self.proj_bn(self.proj_conv(out))
        pe  = self.pe_bn(self.pe_conv(x))
        return layers.Add()([out, pe])


app = FastAPI(
    title="API Clasificación de Corales",
    description="Detecta corales blanqueados a partir de imágenes",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # cambia por tu dominio en prod.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("CORAL_MODEL", "best_Model_Classifier_keras.h5")
try:
    model = load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={"AttentionLayer": AttentionLayer},
    )
except Exception as e:
    raise RuntimeError(f"No se pudo cargar el modelo '{MODEL_PATH}': {e}")

# tamaño de entrada inferido
input_h, input_w = model.input_shape[1:3]
CLASS_NAMES      = [ "Blanqueado", "Saludable"]

def preprocess(img_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((input_w, input_h))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, ...]               # (1, H, W, 3)


@app.get("/")
def health():
    return {"status": "ok", "message": "API de corales funcionando"}

@app.post("/predict")
async def predict(files: list[UploadFile] = File(...)):
    predictions = []
    
    # Procesar cada archivo de imagen
    for file in files:
        if not file.content_type.startswith("image/"):
            raise HTTPException(400, "El archivo debe ser una imagen")

        img_bytes = await file.read()
        try:
            preds = model.predict(preprocess(img_bytes))
            class_idx = np.argmax(preds, axis=1)[0]
            probs  = tf.sigmoid(preds).numpy()
            print(f"Predicción: {probs}, Clase: {class_idx}")
        except Exception as e:
            raise HTTPException(500, f"Error de inferencia para la imagen {file.filename}: {e}")

        if preds.shape[-1] == 1:
            p = float(preds[0][0])
            label = CLASS_NAMES[p >= 0.75]
            confidence = float(tf.sigmoid(p).numpy())
        else:
            label = CLASS_NAMES[class_idx]
            confidence = float(probs[0][class_idx])
        
        # Agregar la predicción a la lista
        predictions.append({
            "filename": file.filename,
            "label": label,
            "confidence": confidence
        })

    return {"predictions": predictions}