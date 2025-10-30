import cv2
from cvzone.HandTrackingModule import HandDetector

# Inisialisasi pengambilan video dari kamera (indeks 0)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka.")

# Inisialisasi HandDetector
detector = HandDetector(staticMode=False, maxHands=1,
                        modelComplexity=1, detectionCon=0.5,
                        minTrackCon=0.5)

# Loop utama untuk memproses frame
while True:
    ok, img = cap.read()
    if not ok: break

    # Temukan tangan dalam frame
    hands, img = detector.findHands(img, draw=True, flipType=True) # flipType untuk mirror UI

    if hands:
        # Ambil data tangan pertama
        hand = hands[0] # dict berisi "lmList", "bbox", dll.

        # Tentukan jari mana yang terangkat
        fingers = detector.fingersUp(hand) # list panjang 5 berisi 0/1

        # Hitung total jari yang terangkat
        count = sum(fingers)

        # Tampilkan jumlah jari dan status fingersUp pada frame
        cv2.putText(img, f"Fingers: {count} {fingers}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Tampilkan frame
    cv2.imshow("Hands + Fingers", img)

    # Keluar jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# Setelah loop berakhir, bebaskan sumber daya dan tutup jendela
cap.release()
cv2.destroyAllWindows()