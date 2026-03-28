# 🛣️ TollAI — Smart Automatic Toll Collection System
**Student:** Yash Patil | Roll No: 24101A0065 | Subject: MVRP

---

## 📁 Project Structure
```
toll_system/
├── app.py                  ← Flask backend (AI + database logic)
├── requirements.txt        ← All Python packages
├── toll.db                 ← SQLite database (auto-created on first run)
├── uploads/                ← Temp folder for videos (auto-created)
└── templates/
    ├── index.html          ← Live Toll Detection Dashboard
    └── admin.html          ← Admin Panel
```

---

## ⚙️ Setup (Do this ONCE)

### Step 1 — Make sure Python 3.9+ is installed
```
python --version
```

### Step 2 — Open terminal in this folder, install packages
```
pip install -r requirements.txt
```
> ⚠️ EasyOCR will download language models (~200MB) on first run. Be patient!

---

## ▶️ Run the App

```
python app.py
```

Then open your browser:
| Page | URL |
|------|-----|
| Live Toll Detection | http://localhost:5000 |
| Admin Panel | http://localhost:5000/admin |

---

## 🧠 How It Works

1. **Upload** a traffic video on the main page
2. **YOLOv8** detects vehicles frame by frame
3. **EasyOCR** reads the number plate from each detected vehicle
4. The plate is **looked up in SQLite** — if registered, toll is deducted
5. If balance is too low → transaction logged as "Insufficient"
6. If plate not found → logged as "Unregistered"
7. **Live feed** shows every transaction as it happens
8. **Admin Panel** lets you manage vehicles and top up balances

---

## 💰 Toll Rates
| Vehicle | Rate |
|---------|------|
| 🚗 Car | ₹30 |
| 🏍️ Motorcycle | ₹15 |
| 🚌 Bus | ₹80 |
| 🚚 Truck | ₹100 |

---

## 🗄️ Pre-loaded Demo Vehicles
The database comes with 8 sample vehicles for demo:
- MH12AB1234 — Rahul Sharma (Car) — ₹800
- MH14CD5678 — Priya Patel (Car) — ₹250
- MH01EF9012 — Suresh Kumar (Truck) — ₹1200
- MH20GH3456 — Anita Desai (Motorcycle) — ₹50 ← LOW
- MH04IJ7890 — Vikram Singh (Bus) — ₹600
- KA05KL2345 — Deepa Nair (Car) — ₹900
- DL08MN6789 — Amit Gupta (Truck) — ₹10 ← CRITICAL
- GJ01OP1234 — Meena Shah (Motorcycle) — ₹300

---

## 🛠️ Tech Stack
| Layer | Technology |
|-------|-----------|
| Vehicle Detection | YOLOv8 (ultralytics) |
| Plate Reading | EasyOCR |
| Video Processing | OpenCV |
| Database | SQLite (built into Python) |
| Backend | Flask |
| Frontend | HTML + CSS + Vanilla JS |
| Real-time Updates | Server-Sent Events (SSE) |

---

## 📌 Tips
- Use a **clear daytime traffic video** for best plate recognition
- EasyOCR works best on **Indian-style plates** (white background, black text)
- If OCR can't read a plate clearly, the vehicle is logged as "Unregistered"
- You can manually add vehicles via the Admin Panel before running detection
