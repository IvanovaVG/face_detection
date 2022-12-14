#from imutils import paths
import face_recognition, cv2, os, pickle, time
from collections import Counter

ti = time.time()
print('[INFO] creating facial embeddings...')

data = pickle.loads(open('./encodings.pickle', 'rb').read())    #encodings here
print('Done! \n[INFO] recognising faces in image...')
imagePaths = './test.jpeg'    #test image here

for (_, imagePath) in enumerate(imagePaths):
    if '_output' not in imagePath:
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model='cnn')    #detection_method here
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []
        for encoding in encodings:
            votes = face_recognition.compare_faces(data['encodings'], encoding)
            if True in votes:
                names.append(Counter([name for name, vote in list(zip(data['names'], votes)) if vote == True]).most_common()[0][0])
            else:
                names.append('Unknown')
        for ((top, right, bottom, left), name) in zip(boxes, names):
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.imwrite(imagePath.rsplit('.', 1)[0] + '_output.jpg', image)

print('Done! \nTime taken: {:.1f} minutes'.format((time.time() - ti)/60))