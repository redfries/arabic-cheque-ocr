# The Uncensored Dev Log: How We Actually Unified the Cheque OCR Pipeline

*Author: Shabaaz Hussain (KFUPM Master's Project)*

Let’s be honest: taking three separate machine learning models trained on different architectures, running them on a single GPU in a serverless environment, and expecting them to cooperate is a recipe for absolute chaos. 

When we started, we had:
1. A **Cascade R-CNN** notebook running on old Detectron2 code.
2. A custom **CRNN + CTC** digits classifier script.
3. A Hugging Face fine-tuning script for **Qwen3.5-0.8B** with LoRA adapters.

They lived in separate folders, had mutually exclusive Python package requirements, and separate weights. Here is the raw, step-by-step developer log of the battles we fought to turn this into a single, cohesive, self-verifying pipeline.

---

## Battle 1: The Dependency Hell (`setuptools` compilation trap)

The first step was trying to run Detectron2 (Cascade R-CNN) and Qwen3.5 (Hugging Face Transformers + PEFT) in the same script. We immediately ran into a wall of compiler errors. 

### The Problem:
* Detectron2’s compilation scripts rely on old Python installation utilities (specifically, calling deprecated modules inside `distutils`).
* Modern Hugging Face packages require a brand new Python ecosystem.
* If you run a standard `pip install -r requirements.txt` on a modern system, Python upgrades `setuptools` to version 70+.
* The moment `setuptools>=70` is active, running `pip install detectron2` fails with a cryptic compilation error: `ModuleNotFoundError: No module named 'distutils'`.

### The Hack:
We had to build a custom build sequence for our environments (especially inside the Modal serverless container). We couldn't install them in random order. We had to execute:
1. **Force downgrading setuptools**: Run `pip install 'setuptools<70'` before doing *anything* else.
2. **Pre-installing PyTorch matching CUDA wheels**: Run `pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121` so the system has PyTorch with CUDA support pre-compiled.
3. **Compile Detectron2 from source**: Now we compile Detectron2 using the pinned setuptools. It successfully finds the CUDA compilation libraries and builds.
4. **Install modern packages**: Finally, run `pip install transformers peft qwen-vl-utils streamlit opencv-python`.

This sequencing bridged what were originally two completely isolated Python environments into a single, unified runtime.

---

## Battle 2: The GPU Context & CUDA OOM Battles

Once the libraries were installed, we put them all in one Python script: load Detector predictor, load CRNN model, load Qwen3.5 Vision model. We pressed run, and the terminal crashed with:
`torch.cuda.OutOfMemoryError: CUDA out of memory.`

### The Problem:
We had three heavy models sharing the same GPU memory space inside the same process:
* Cascade R-CNN (ResNet-50 back-bone) takes ~1.5GB of GPU VRAM.
* CRNN takes a small footprint (~200MB).
* Qwen3.5-0.8B is a Vision-Language transformer model. If loaded in standard `float32` precision, it consumes over 3.5GB of VRAM just to load weights, plus a dynamic key-value (KV) cache that expands when processing images.
* Adding the Streamlit web server overhead pushed the VRAM usage past the limits of standard low-end GPUs.

### The Solution:
We had to aggressively optimize memory loading inside the unified pipeline class `ChequeOCRPipeline`:
1. **bfloat16 Cast**: We detected if CUDA is available and loaded the Qwen3.5 base model using `torch.bfloat16` precision:
   ```python
   dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
   ```
   This immediately cut Qwen's model weights memory usage in half (~1.6GB).
2. **HF Device Map Orchestration**: We configured Qwen with `device_map="auto"` and `trust_remote_code=True`. This allowed Hugging Face to automatically slice the model weights and load them efficiently, while we manually assigned the CRNN and Cascade R-CNN models to the explicit GPU device context.
3. **Garbage Collection**: We wrapped the inference predictions inside `with torch.no_grad():` blocks to prevent PyTorch from keeping activation gradients in memory.

---

## Battle 3: The Streamlit TIFF Crash

When we ran the web app locally or on Modal, we uploaded a cheque image in `.tif` (TIFF) format. The page immediately broke, displaying a blank screen with a red error block.

### The Problem:
The cheque dataset contains high-quality `.tif` files. Web browsers (Chrome, Safari, Firefox) and Streamlit's standard image widget `st.image` **cannot render TIFF files natively**. The browser simply doesn't have the decoder for it, leading to broken page rendering.

### The Solution:
We had to intercept the image load process. Instead of passing the file path or raw bytes directly to Streamlit, we built a robust decoding pipeline:
1. We read the TIFF image using OpenCV (`cv2.imread`) or fallback to Pillow (`Image.open`).
2. We convert the channel format from BGR (OpenCV standard) to RGB.
3. We compress the raw image array into a web-friendly PNG format in-memory using `cv2.imencode('.png', img)`.
4. We base64-encode the resulting bytes and format it as an inline HTML data URI:
   ```python
   import base64
   _, buffer = cv2.imencode('.png', image_rgb)
   img_base64 = base64.b64encode(buffer).decode('utf-8')
   data_url = f"data:image/png;base64,{img_base64}"
   ```
5. We render this `data_url` in Streamlit. This guarantees that every image loads perfectly in any web browser.

---

## Battle 4: The Bounding Box & Guideline Nightmare

When the Cascade R-CNN detector localized the courtesy digits box, the crop was often slightly offset or included noise.

```text
[ Raw Box Crop ]  --> [ Includes dotted lines + border ] --> [ OCR reads "1 5 0 0 -" ]
```

### The Problem:
Cheques have vertical and horizontal dotted lines for writing amounts. When the detector cropped the box, the crop included these black line strokes. The CRNN model would misread a guideline stroke as a digit `1` or a noise dash `-`, leading to false courtesy amounts (e.g. reading `5150` as `15150`).

### The Solution:
We implemented an image-processing pipeline using morphological filtering on the cropped gray-scale image:
1. **Border Whitening**: We set the outermost 3% border of the cropped box to solid white (`255`) to eliminate bounding box outline pieces.
2. **Line Removal via Opening**: We applied OpenCV morphological opening. We generated horizontal and vertical structuring kernels and subtracted the detected lines from the image:
   ```python
   hlines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, h_kernel)
   vlines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, v_kernel)
   lines = cv2.bitwise_or(hlines, vlines)
   out[lines > 0] = 255  # Clean guidelines out of the crop
   ```
3. **Safety Fallback**: If the morphological line removal erased too much ink (which we check by measuring the remaining ink ratio), the pipeline automatically rolls back to the clean bordered crop.

---

## Battle 5: The Spelling Mistake & Translation Gap

Qwen3.5 reads the written legal amount text. But handwritten words have spelling mistakes (e.g., writing `ثلاتين` instead of `ثلاثين`). Furthermore, we had to check if these words match the digits.

### The Problem:
How do you match `"خمسة آلاف ومائة وخمسون"` against `"5150"` programmatically?
A standard lookup table fails because of cursive handwriting mistakes, regional word variations (e.g., `"مئة"`, `"مية"`, `"مائة"` all mean 100), and grammar connections like the Arabic conjunction letter `"و"` (and) merged with words.

### The Solution:
We built a custom parser (`parse_legal_amount_v2_fix2`) executing a four-step pipeline:
1. **Orthographic Normalization**: Normalizes letters (Alif, Yaa, diacritics, elongation tatweel).
2. **De-Noising**: Stretches and splits conjunctions like `"و"` and splits compound terms like `"عشرالف"` $\rightarrow$ `["عشرة", "الف"]`.
3. **Adaptive Levenshtein Edit-Distance**: Instead of doing an exact string match on number terms (like `"واحد"`, `"ثلاثون"`), the parser loops over a dictionary of ~60 valid Arabic number words and calculates the Levenshtein distance.
   * If a word is short ( $\leq 3$ characters), we enforce a strict threshold (distance $\leq 1$, ratio $\leq 0.34$).
   * If a word is long ( $\leq 6$ characters), we allow distance $\leq 2$.
   * This successfully corrects spelling mistakes without causing false-positives (e.g. mistaking `عشر` (10) for `عشرون` (20)).
4. **Grammar Accumulation**: Loops through the corrected words, multiplying coefficients when encountering multipliers like `"الف"` (1000) or `"مليون"` (1000000) to yield the final parsed float amount.

---

## Battle 6: Serverless Cold Start Optimization

Deploying the pipeline on serverless container infrastructure (Modal.com) introduced latency issues.

### The Problem:
* A serverless container starts from a cold state.
* If the container has to fetch Qwen3.5 base weights (1.6GB) and detector parameters (300MB) from Hugging Face or an external URL on every cold start, the boot time takes **over 45 seconds**.
* The container will time out, and the web client will disconnect.

### The Solution:
We used a **Modal Persistent Volume** to build a local network-attached flash storage system:
1. We created a volume named `cheque-ocr-models`.
2. We wrote a one-time setup script (`setup_models.py`) that pre-downloaded the base models and LoRA adapters directly to the local folder, then pushed them to the Modal volume.
3. In `modal_app.py`, we mount the volume to `/root/models` inside the container:
   ```python
   @app.function(
       gpu="A10G",
       volumes={"/root/models": volume},
       image=image
   )
   ```
4. Inside our unified pipeline script, we resolve paths dynamically: if `/root/models` exists (running on Modal), we load directly from flash storage. Otherwise, we fall back to the local relative path.
5. This dropped the pipeline startup time to **under 3 seconds** because the weights are cached locally inside the serverless infrastructure.

---

## The Unified Execution Flow

Here is the raw execution loop inside `process_one` in `pipeline_core.py` showing how they cooperate:

```python
# 1. Run Detector
det_out = self.predictor(bgr_det)

# 2. Crop with pad scaling
courtesy_box = pick_top_box(inst, courtesy_class)
legal_box = pick_top_box(inst, legal_class)

# 3. Clean and Run Courtesy Digit OCR
stages, enh_rs, dbg = preprocess_crop_array(courtesy_crop_gray)
ocr_digits = self.ocr_model(enh_rs) # CTC Decoded -> e.g. "5150"

# 4. Run Legal VL OCR
ocr_legal_text = run_qwen_ocr(legal_crop_rgb) # Transcribes text

# 5. Parse Legal Text
parsed_result = parse_legal_amount_v2_fix2(ocr_legal_text) # Returns float e.g. 5150

# 6. Verify and Fallback Loop
if parsed_result["amount"] == int(ocr_digits):
    verified = True
else:
    # Mismatch! Run fallback enhancements on the crop and try again
    enhanced_crop = enhance_border_auto(legal_crop_rgb)
    fallback_text = run_qwen_ocr(enhanced_crop)
    if parse_legal_amount_v2_fix2(fallback_text)["amount"] == int(ocr_digits):
        ocr_legal_text = fallback_text
        verified = True
```

This cooperative design ensures that the fast digit model and the semantic language model check each other's work, providing a high-confidence, verified classification result.
