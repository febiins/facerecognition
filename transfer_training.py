import os
import numpy as np
from deepface import DeepFace

dataset_path = "dataset"

embeddings = []
labels = []

print("[INFO] Training using FaceNet (Transfer Learning)...")

for person_name in os.listdir(dataset_path):

    person_folder = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_folder):
        continue

    for image_name in os.listdir(person_folder):

        image_path = os.path.join(person_folder, image_name)

        try:
            result = DeepFace.represent(
                img_path=image_path,
                model_name="Facenet",
                detector_backend="opencv",
                enforce_detection=False
            )

            embedding = result[0]["embedding"]

            embeddings.append(embedding)
            labels.append(person_name)

            print(f"[INFO] Processed {image_name} for {person_name}")

        except Exception as e:
            print(f"[WARNING] Skipped {image_name}: {e}")

embeddings = np.array(embeddings)
labels = np.array(labels)

np.save("embeddings.npy", embeddings)
np.save("labels.npy", labels)

print("\n[INFO] Training completed successfully.")
print(f"[INFO] Total persons trained: {len(set(labels))}")