# Ensemble: Bagging vs Boosting

เปรียบเทียบ Bagging (Parallel / Sequential) กับ Boosting (SAMME) ที่เขียนเองด้วย NumPy
บนข้อมูล Iris โดยใช้ Decision Tree เป็น base learner

## โครงสร้าง

| ไฟล์ | หน้าที่ |
|---|---|
| `tree.py` | Decision Tree (Gini) ใช้เป็น base learner |
| `bagging.py` | `SequentialBagging` และ `ParallelBagging` (multi-process) |
| `boosting.py` | `SequentialBoosting` (AdaBoost แบบหลายคลาส) |
| `compare.py` | ชุดทดลอง + วาดกราฟ |

## วิธีรัน

```bash
pip install -r requirements.txt
python compare.py
```

ได้กราฟ `results.png`

## การทดลอง

1. **Accuracy vs ความลึกต้นไม้** (5-fold x 3 รอบ)
2. **Accuracy vs label noise** (สลับ label เฉพาะชุด train, depth=3)
3. **เวลาเทรน** Bagging sequential vs parallel vs Boosting ตามขนาดข้อมูล

![results](results.png)

## ข้อสังเกต

- บนข้อมูลสะอาด Bagging และ Boosting แม่นใกล้เคียงกัน
- เมื่อมี label noise Boosting ตกเร็วกว่า เพราะเพิ่ม weight ให้ข้อมูลที่ทายผิด (รวมถึง label ที่ผิดจริง)
- Bagging ไม่ลด bias จึงแย่มากเมื่อใช้ stump (depth=1)
- Parallel คุ้มเมื่อข้อมูลใหญ่และเครื่องมีหลายคอร์ ข้อมูลเล็กจะเสียเวลาสร้าง process