import cv2
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector

# Indeks landmark mata kiri (berdasarkan MediaPipe Face Mesh)
# L_TOP (159): atas, L_BOTTOM (145): bawah
# L_LEFT (33): kiri, L_RIGHT (133): kanan
L_TOP, L_BOTTOM, L_LEFT, L_RIGHT = 159, 145, 33, 133

# Fungsi untuk menghitung jarak Euclidean antara dua titik (p1, p2)
def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Inisialisasi pengambilan video dari kamera (indeks 0)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak bisa dibuka.")

# Inisialisasi objek FaceMeshDetector
detector = FaceMeshDetector(staticMode=False, maxFaces=2,
                            minDetectionCon=0.5, minTrackCon=0.5)

# Variabel untuk menghitung kedipan sederhana
blink_count = 0
closed_frames = 0
CLOSED_FRAMES_THRESHOLD = 3  # Jumlah frame berturut-turut untuk dianggap kedipan
EYE_AR_THRESHOLD = 0.20      # Ambang Eye Aspect Ratio (EAR) untuk menilai mata tertutup
is_closed = False            # Status apakah mata sudah dihitung sebagai 'tertutup'

# Loop utama untuk memproses frame
while True:
    ok, img = cap.read()
    if not ok: break

    # Temukan Face Mesh di dalam frame
    img, faces = detector.findFaceMesh(img, draw=True)

    if faces:
        # Ambil data mesh wajah pertama
        face = faces[0]  # list of 468 (x,y) landmarks

        # Hitung jarak vertikal (v) dan horizontal (h) mata kiri
        v = dist(face[L_TOP], face[L_BOTTOM])
        h = dist(face[L_LEFT], face[L_RIGHT])

        # Hitung Eye Aspect Ratio (EAR)
        # Menambahkan 1e-8 untuk menghindari pembagian dengan nol
        ear = v / (h + 1e-8)

        # Tampilkan nilai EAR pada frame
        cv2.putText(img, f"EAR(L): {ear:.3f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # Logika counter kedipan sederhana:
        if ear < EYE_AR_THRESHOLD:
            closed_frames += 1

            # Jika mata tertutup cukup lama dan belum dihitung sebagai kedipan
            if closed_frames >= CLOSED_FRAMES_THRESHOLD and not is_closed:
                blink_count += 1
                is_closed = True  # Tandai mata sudah dihitung sebagai tertutup/kedipan
        else:
            # Jika mata terbuka (EAR > EYE_AR_THRESHOLD)
            closed_frames = 0
            is_closed = False  # Setel ulang status

        # Tampilkan jumlah kedipan pada frame
        cv2.putText(img, f"Blink: {blink_count}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Tampilkan frame
    cv2.imshow("FaceMesh + EAR", img)

    # Keluar jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# Setelah loop berakhir, bebaskan sumber daya dan tutup jendela
cap.release()
cv2.destroyAllWindows()