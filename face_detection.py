import cv2
import numpy as npy
import face_recognition as face_rec
def resize(img, size) :
    width = int(img.shape[1]*size)
    height = int(img.shape[0] *size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)




# img declaration
shyda = face_rec.load_image_file('sample_image\shyda.jpg')
shyda = cv2.cvtColor(shyda, cv2.COLOR_BGR2RGB)
shyda = resize(shyda, 0.50)
shyda_test = face_rec.load_image_file('sample_image\elon_musk.jpg')
shyda_test = resize(shyda_test, 0.50)
shyda_test = cv2.cvtColor(shyda_test, cv2.COLOR_BGR2RGB)

# finding face location
faceLocation_shyda = face_rec.face_locations(shyda)[0]
encode_shyda = face_rec.face_encodings(shyda)[0]
cv2.rectangle(shyda, (faceLocation_shyda[3], faceLocation_shyda[0]), (faceLocation_shyda[1], faceLocation_shyda[2]), (255, 0, 255), 3)


faceLocation_shydatest = face_rec.face_locations(shyda_test)[0]
encode_shydatest = face_rec.face_encodings(shyda_test)[0]
cv2.rectangle(shyda_test, (faceLocation_shyda[3], faceLocation_shyda[0]), (faceLocation_shyda[1], faceLocation_shyda[2]), (255, 0, 255), 3)


results = face_rec.compare_faces([encode_shyda], encode_shydatest)
print(results)
cv2.putText(shyda_test, f'{results}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255), 2)

cv2.imshow('main_img', shyda)
cv2.imshow('test_img', shyda_test)
cv2.waitKey(0)
cv2.destroyAllWindows()
