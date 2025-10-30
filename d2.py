import cv2 
from cvzone.PoseModule import PoseDetector

# Inisialisasi pengambilan video dari kamera (indeks 0)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    # Munculkan kesalahan jika kamera tidak dapat dibuka
    raise RuntimeError("Kamera tidak bisa dibuka.")

# Inisialisasi PoseDetector
detector = PoseDetector(staticMode=False, modelComplexity=1,
                        enableSegmentation=False, detectionCon=0.5,
                        trackCon=0.5)

# Loop utama untuk memproses frame
while True:
    # Tangkap setiap frame dari webcam
    success, img = cap.read()
 
    # Periksa apakah frame berhasil ditangkap
    if not success:
        print("Gagal membaca frame dari kamera.")
        break

    # Temukan pose manusia dalam frame
    img = detector.findPose(img)

    # Temukan landmark (lmList) dan info bounding box (bboxInfo)
    lmList, bboxInfo = detector.findPosition(img, draw=True,
                                             bboxWithHands=False)

    # Periksa apakah ada landmark tubuh yang terdeteksi
    if lmList:
        # Dapatkan pusat bounding box
        center = bboxInfo["center"]

        # Gambar lingkaran di pusat bounding box
        cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

        # Hitung jarak antara landmark 11 (Bahu Kiri) dan 15 (Pergelangan Tangan Kiri)
        length, img, info = detector.findDistance(lmList[11][0:2],
                                                  lmList[15][0:2],
                                                  img=img,
                                                  color=(255, 0, 0),
                                                  scale=10)
                                                  
        print("Jarak: ", length)
        # Hitung sudut antara landmark 11 (Bahu Kiri), 13 (Siku Kiri), dan 15 (Pergelangan Tangan Kiri)
        angle, img = detector.findAngle(lmList[11][0:2],
                                        lmList[13][0:2],
                                        lmList[15][0:2],
                                        img=img,
                                        color=(0, 0, 255),
                                        scale=10)

        # Periksa apakah sudut mendekati 50 derajat dengan toleransi 10
        isCloseAngle50 = detector.angleCheck(myAngle=angle,
                                             targetAngle=50,
                                             offset=10)

        # Cetak hasil pemeriksaan sudut
        print(isCloseAngle50)

    # Tampilkan frame
    cv2.imshow("Pose + Angle ", img)

    # Keluar jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bebaskan sumber daya dan tutup jendela
cap.release()
cv2.destroyAllWindows()