import cv2

from ultralytics import YOLO


class AIDetector:

    def __init__(self, model_path, confidence=0.5):

        self.model = YOLO(model_path)

        self.confidence = confidence

    def detect(self, frame):

        results = self.model(
            frame,
            conf=self.confidence,
            verbose=False
        )

        detections = []

        result = results[0]

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            class_name = self.model.names[class_id]

            detections.append({
                "class": class_name,
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2]
            })

        return detections

    def draw_detections(self, frame, results):

        output = frame.copy()

        for detection in results:

            x1, y1, x2, y2 = map(
                int,
                detection["bbox"]
            )

            label = (
                f'{detection["class"]} '
                f'{detection["confidence"]:.2f}'
            )

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                output,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        return output