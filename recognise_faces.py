import face_recognition
import pickle
import cv2
import os
from collections import Counter
import imutils


class ModelCNN:
    def __init__(self):
        self.encodings = './encodings.pickle'
        self.detection_method = 'cnn'
        self.data = pickle.loads(open(self.encodings, 'rb').read())

    def recognise_faces(self, image_detect):
        # load the known faces and embeddings
        print('loading encodings...')
        # data = pickle.loads(open(self.encodings, 'rb').read())
        # load the input image and convert it from BGR to RGB
        image = cv2.imread(image_detect)
        image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # detect the (x,)-coordinates of the bounding boxes corresponding to each face in the input image, then compute the facial embeddings for each face
        boxes = face_recognition.face_locations(rgb, model=self.detection_method)
        encodings = face_recognition.face_encodings(rgb, boxes)
        # initialize the list of names for each face detected
        names = []

        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known encodings, function returns a list of True/False values, one for each known encoding
            # Internally, the compare_faces function is computing the Euclidean distance between the candidate embedding and all faces in our known encodings
            votes = face_recognition.compare_faces(self.data['encodings'], encoding)
            # check to see if a match is found
            if True in votes:
                # find the corresponding names of all faces matched (vote==True)
                matches = [name for name, vote in list(zip(self.data['names'], votes)) if vote == True]
                # determine the most frequently occuring name (note: in the unlikely event of a tie, Python will select first entry in the dictionary)
                name = Counter(matches).most_common()[0][0]
            else:
                name = 'Unknown'
            # update the list of names
            names.append(name)
        # visualise with bounding boxes and labeled names, loop over the recognised faces
        print(boxes)
        for ((top, right, bottom, left), name) in zip(boxes, names):
            scale_w = image.shape[1] / 112
            scale_h = image.shape[0] / 112
            print(image.shape[1], image.shape[0])
            # draw the predicted face name on the image
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            #y = top - 15 if top - 15 > 15 else top + 15
            #cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Display the resulting frame, press 'q' to exit
        window_text = image_detect.split(os.path.sep)[-1]
        cv2.imshow(window_text, image)
        # Save output image
        cv2.imwrite(image_detect.rsplit('.', 1)[0] + '_output.jpg', image)


# args['image'] = './test.jpeg'
# args['detection_method'] = 'cnn'
# recognise_faces(args)
model = ModelCNN()
model.recognise_faces('./new.jpeg')