import cv2
import os

def extract_frames(video_path, output_folder, interval):
    # Otwórz plik wideo
    cap = cv2.VideoCapture(video_path)

    # Upewnij się, że plik wideo został poprawnie otwarty
    if not cap.isOpened():
        print("Błąd: Nie udało się otworzyć pliku wideo.")
        return

    # Utwórz folder wyjściowy, jeśli nie istnieje
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Zmienna do śledzenia numeru klatki
    frame_number = 0

    while True:
        # Odczytaj klatkę
        ret, frame = cap.read()

        # Przerwij pętlę, gdy skończy się odczyt klatek
        if not ret:
            break

        # Sprawdź, czy numer klatki jest podzielny przez interval
        if frame_number % interval == 0:
            # Zapisz klatkę jako obraz
            frame_filename = os.path.join(output_folder, f"frame_{frame_number:04d}.jpg")
            cv2.imwrite(frame_filename, frame)

        # Zwiększ numer klatki
        frame_number += 1

    # Zamknij plik wideo
    cap.release()

if __name__ == "__main__":
    # Ścieżka do pliku wideo
    video_path = "film/try.mp4"

    # Katalog, w którym zostaną zapisane klatki
    output_folder = "zdjecia"

    extract_frames(video_path, output_folder, 10)
