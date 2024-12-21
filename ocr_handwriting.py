# USAGE
# python ocr_handwriting.py --model handwriting.model --image images/umbc_address.png
from difflib import SequenceMatcher

# import the necessary packages
from tensorflow.keras.models import load_model
from imutils.contours import sort_contours
import numpy as np
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser(description="OCR Handwriting Evaluation")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to trained handwriting recognition model")
args = vars(ap.parse_args())

# load the handwriting OCR model
print("[INFO] loading handwriting OCR model...")
model = load_model(args["model"])

# load the input image from disk, convert it to grayscale, and blur
# it to reduce noise
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# perform edge detection, find contours in the edge map, and sort the
# resulting contours from left-to-right
edged = cv2.Canny(blurred, 30, 150)
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method="left-to-right")[0]

# initialize the list of contour bounding boxes and associated
# characters that we'll be OCR'ing
chars = []

# loop over the contours
for c in cnts:
	# compute the bounding box of the contour
	(x, y, w, h) = cv2.boundingRect(c)

	# filter out bounding boxes, ensuring they are neither too small
	# nor too large
	if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
		# extract the character and threshold it to make the character
		# appear as *white* (foreground) on a *black* background, then
		# grab the width and height of the thresholded image
		roi = gray[y:y + h, x:x + w]
		thresh = cv2.threshold(roi, 0, 255,
			cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
		(tH, tW) = thresh.shape

		# if the width is greater than the height, resize along the
		# width dimension
		if tW > tH:
			thresh = imutils.resize(thresh, width=32)

		# otherwise, resize along the height
		else:
			thresh = imutils.resize(thresh, height=32)

		# re-grab the image dimensions (now that its been resized)
		# and then determine how much we need to pad the width and
		# height such that our image will be 32x32
		(tH, tW) = thresh.shape
		dX = int(max(0, 32 - tW) / 2.0)
		dY = int(max(0, 32 - tH) / 2.0)

		# pad the image and force 32x32 dimensions
		padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
			left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
			value=(0, 0, 0))
		padded = cv2.resize(padded, (32, 32))

		# prepare the padded image for classification via our
		# handwriting OCR model
		padded = padded.astype("float32") / 255.0
		padded = np.expand_dims(padded, axis=-1)

		# update our list of characters that will be OCR'd
		chars.append((padded, (x, y, w, h)))

# extract the bounding box locations and padded characters
boxes = [b[1] for b in chars]
chars = np.array([c[0] for c in chars], dtype="float32")

# OCR the characters using our handwriting recognition model
preds = model.predict(chars)

# define the list of label names
labelNames = "0123456789"
labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames = [l for l in labelNames]

ground_truth = "HWELOLROLD"  # Chuỗi mẫu, thay đổi tùy vào dữ liệu thực tế
# Khởi tạo danh sách lưu kết quả và chuỗi dự đoán
results = []
predicted_text = ""
# loop over the predictions and bounding box locations together
for (pred, (x, y, w, h)) in zip(preds, boxes):
	# find the index of the label with the largest corresponding
	# probability, then extract the probability and label
	i = np.argmax(pred)
	prob = pred[i]
	label = labelNames[i]

	# Lưu kết quả vào danh sách
	results.append({
		"character": label,
		"probability": prob,
		"bounding_box": (x, y, w, h),
	})

	# Xây dựng chuỗi dự đoán
	predicted_text += label

	# draw the prediction on the image
	print("[INFO] {} - {:.2f}%".format(label, prob * 100))
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
	cv2.putText(image, label, (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

	# show the image
	cv2.imshow("Image", image)
	cv2.waitKey(0)

# Tính các chỉ số đánh giá
correct_chars = sum(1 for gt, pred in zip(ground_truth, predicted_text) if gt == pred)
char_accuracy = correct_chars / len(ground_truth)
sequence_accuracy = 1 if ground_truth == predicted_text else 0
average_probability = np.mean([result['probability'] for result in results])
similarity = SequenceMatcher(None, ground_truth, predicted_text).ratio()

# Hiển thị kết quả đánh giá
print("[EVALUATION] Character Accuracy: {:.2f}%".format(char_accuracy * 100))
print("[EVALUATION] Sequence Accuracy: {:.2f}%".format(sequence_accuracy * 100))
print("[EVALUATION] Average Confidence: {:.2f}%".format(average_probability * 100))
print("[EVALUATION] Similarity Score: {:.2f}%".format(similarity * 100))

# Vẽ biểu đồ đánh giá
evaluation_metrics = ['Character Accuracy', 'Sequence Accuracy', 'Average Confidence', 'Similarity Score']
values = [char_accuracy * 100, sequence_accuracy * 100, average_probability * 100, similarity * 100]

plt.figure(figsize=(10, 6))
plt.bar(evaluation_metrics, values, color=['blue', 'orange', 'green', 'red'])
plt.ylim(0, 100)
plt.title('OCR Evaluation Metrics')
plt.ylabel('Percentage (%)')
plt.xlabel('Metrics')
plt.xticks(rotation=45)
for i, v in enumerate(values):
    plt.text(i, v + 2, f"{v:.2f}%", ha='center', fontsize=10)

plt.tight_layout()
plt.show()