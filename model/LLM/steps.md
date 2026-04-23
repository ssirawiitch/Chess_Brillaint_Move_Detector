# Chess Brilliant Move Predictor
---

## Phase 1: Setup & Data Extraction (สกัดข้อมูล)

**เป้าหมาย:** อ่านไฟล์ PGN แล้วคำนวณฟีเจอร์ระดับเทพ (Material Diff, Delta Eval)

* **Step 1: สร้างโครงโปรเจกต์**
  * **Prompt ให้ AI:** > "กำลังจะทำโปรเจกต์ AI ทำนาย Brilliant Move หมากรุก ช่วยสร้างไฟล์ `requirements.txt` ที่มี `python-chess`, `pandas`, `datasets`, `transformers`, `peft`, `trl`, `bitsandbytes` ให้หน่อย และสร้างโฟลเดอร์โครงสร้างรอไว้เลย (มีโฟลเดอร์ data/, scripts/, models/)"

* **Step 2: เขียนสคริปต์อ่าน PGN และคำนวณฟีเจอร์**
  * **Prompt ให้ AI:**
    > "เขียนสคริปต์ `scripts/extract_features.py` โดยใช้ `python-chess` อ่านไฟล์ PGN. สำหรับแต่ละตาเดิน ให้ดึงข้อมูลดังนี้: 1. FEN ของกระดานก่อนเดิน 2. ตาเดินนั้น 3. is_check 4. material_diff (นับมูลค่าหมากที่ต่างกัน) 5. delta_eval (ถ้าในไฟล์มี) 6. is_brilliant (เป็น 1 ถ้ามี '!!' นอกนั้น 0). รวบรวมข้อมูลทั้งหมดใส่ Pandas DataFrame แล้วเซฟเป็น `data/raw_features.csv`"

---

## Phase 2: Balancing & Formatting (จัดการ Imbalance & JSONL)

**เป้าหมาย:** สุ่มลดจำนวนตาเดินปกติ (Downsample) และแปลงตารางเป็นข้อความให้ LLM อ่าน

* **Step 3: ทำ Downsampling**
  * **Prompt ให้ AI:**
    > "เขียนสคริปต์ `scripts/prepare_jsonl.py` โหลดไฟล์ `raw_features.csv` ตอนนี้ข้อมูล Imbalance มาก ให้ทำ Downsampling โดยเก็บแถวที่ is_brilliant=1 ไว้ทั้งหมด แล้วสุ่มแถว is_brilliant=0 มาให้มีจำนวนเป็น 2 เท่าของ brilliant move เพื่อให้สัดส่วนเป็น 1:2"

* **Step 4: แปลงเป็น JSONL Prompt**
  * **Prompt ให้ AI:**
    > "จาก DataFrame ที่ Balance แล้วในโค้ดเมื่อกี้ ช่วยเขียน Loop แปลงแต่ละแถวให้เป็นไฟล์ JSONL (`data/train.jsonl` และ `data/test.jsonl`) โดยให้มี key 'text' และ value มีหน้าตาโครงสร้างแบบนี้เป๊ะๆ: 
    > `Board: {FEN} | Move: {Move} | Check: {is_check} | Material Diff: {material_diff} | Delta Eval: {delta_eval} | Label: {Brilliant หรือ Normal}`"

---

## Phase 3: Model Fine-Tuning (เทรน Qwen ด้วย LoRA)

**เป้าหมาย:** สอน Qwen-2.5-1.5B ให้จับแพทเทิร์น Brilliant Move

* **Step 5: สร้างสคริปต์เทรนโมเดล**
  * **Prompt ให้ AI:**
    > "เขียนสคริปต์ `scripts/train_lora.py` เพื่อ Fine-tune โมเดล 'Qwen/Qwen2.5-1.5B-Instruct' ด้วยข้อมูล `data/train.jsonl`. 
    > ข้อกำหนด: 1. โหลดโมเดลแบบ 4-bit (bitsandbytes) 2. ใช้ LoRA Config (r=16, alpha=32, target_modules=['q_proj', 'v_proj']) 3. ใช้ SFTTrainer จาก `trl` library. 4. เซฟ LoRA Adapter ไว้ที่โฟลเดอร์ `models/chess-lora` 5. ตั้งค่า num_train_epochs=5"

---

## Phase 4: Inference (การนำไปใช้งานจริง)

**เป้าหมาย:** ลองของจริง! เอาโมเดลที่เทรนเสร็จมาทายผลกระดานใหม่

* **Step 6: สร้างสคริปต์ทดสอบ**
  * **Prompt ให้ AI:**
    > "เขียนสคริปต์ `scripts/predict.py` เพื่อใช้ทดสอบโมเดล ให้โหลด Base Model (Qwen2.5) แล้วประกอบร่างกับ LoRA Adapter ใน `models/chess-lora` จากนั้นสร้างฟังก์ชันรับค่าพารามิเตอร์ (FEN, Move, Material Diff, ฯลฯ) จัดฟอร์แมตให้อยู่ในรูป Prompt เหมือนตอนเทรน แล้วสั่งให้โมเดล Generate คำตอบ (Label) ออกมาว่า Brilliant หรือ Normal"

---

## ทำต่อ

หากต้องการนำโค้ดในโปรเจกต์นี้ไปพัฒนาต่อ หรือเทรนโมเดลเพิ่มเติม สามารถทำตามขั้นตอนต่อไปนี้:

1. **Clone/Copy Project**: นำโปรเจกต์นี้ไปรันในเครื่องเป้าหมาย (โฟลเดอร์ที่หนักๆ เช่น `models/` และข้อมูลดิบใน `data/` จะถูกละเว้นด้วย `.gitignore` โดยอัตโนมัติเพื่อให้แชร์โค้ดง่ายขึ้น)
2. **ติดตั้ง Environment**:
   ```bash
   pip install -r requirements.txt
   ```
3. **เตรียมข้อมูล (Data Preparation)**:
   - นำไฟล์ PGN หมากรุกไปรันสกัดฟีเจอร์ด้วย `python scripts/extract_features.py` (หรือถ้ามี `data/raw_features.csv` อยู่แล้ว ข้ามไปข้อถัดไปได้เลย)
   - สุ่มข้อมูลและเตรียมเป็นรูปแบบ JSONL ด้วยคำสั่ง `python scripts/prepare_jsonl.py` (ระบบจะจัดสัดส่วน Brilliant:Normal เป็น 1:2 ให้ตามที่อัปเดตล่าสุด)
4. **เริ่มเทรนโมเดล (Training)**:
   - รันคำสั่ง `python scripts/train_lora.py` 
   - ระบบจะเทรนทั้งหมด 5 Epochs (สามารถเข้าไปแก้ตัวเลขในสคริปต์ได้ถ้าต้องการเพิ่ม/ลด)
5. **ทดสอบผลลัพธ์ (Inference & Evaluation)**:
   - รัน `python scripts/predict.py` เพื่อลองทำนาย
   - รัน `python scripts/evaluate.py` เพื่อดูค่าประสิทธิภาพของโมเดลที่เทรนมา (Precision, Recall, F1)

โค้ดทั้งหมดถูกเขียนให้รองรับการทำงานแบบแยกส่วนกันชัดเจน หากเพื่อนต้องการปรับจูน (เช่น เปลี่ยนอัตราส่วนข้อมูล, เพิ่ม Epoch) สามารถเข้าไปแก้ในไฟล์ `prepare_jsonl.py` และ `train_lora.py` ได้โดยตรงครับ!