import cv2
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

orb = cv2.ORB_create()
def extract_features(images):

    keypoints_list = []
    descriptors_list = []

    for image in images:
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        keypoints, descriptors = orb.detectAndCompute(gray, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)

        # Dibujar los puntos clave en la imagen
        img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=0)

        # Mostrar la imagen con los puntos clave
        cv2.imshow(f'Imagen {image}', img_with_keypoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Imprimir las características de la imagen
        print(f"Características de la imagen {image}:")
        for i, keypoint in enumerate(keypoints):
            print(f"Característica {i+1}:")
            print("Punto: ({}, {})".format(keypoint.pt[0], keypoint.pt[1]))
            print("Tamaño: {}".format(keypoint.size))
            print("Ángulo: {}".format(keypoint.angle))
            print("Respuesta: {}".format(keypoint.response))
            print("Octava: {}".format(keypoint.octave))
            print("ID de clase: {}".format(keypoint.class_id))
            print("------------")  

    return keypoints_list, descriptors_list

def add_images(image_paths, new_images):
    all_images = image_paths + new_images
    keypoints, descriptors = extract_features(all_images)

    data_matrix = []
    labels = []

    for i, keypoints_image in enumerate(keypoints):
        for keypoint in keypoints_image:
            data_matrix.append([
                keypoint.pt[0],
                keypoint.pt[1],
                keypoint.size,
                keypoint.angle,
                keypoint.response,
                keypoint.octave,
                keypoint.class_id
            ])
            labels.append(i)

    data_matrix = np.array(data_matrix)
    labels = np.array(labels)

    return data_matrix, labels

# Lista de rutas de imágenes existentes
image_paths = ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg']

# Nuevas imágenes para comprobar el rendimiento
new_images = ['6.jpg','7.jpg','8.jpg','9.jpg','10.jpg']


# Extraer características y actualizar la matriz de características y etiquetas
data_matrix, labels = add_images(image_paths, new_images)

# Crear el clasificador SVM
svm_classifier = SVC()

# Entrenar el clasificador
svm_classifier.fit(data_matrix, labels)

# Evaluar el rendimiento con validación cruzada
scores = cross_val_score(svm_classifier, data_matrix, labels, cv=5)  # 5-fold cross-validation

# Imprimir la matriz de características
print("Matriz de características:") 
print(data_matrix)

# Imprimir los resultados de validación cruzada
print("Resultados de validación cruzada:")
print(scores)
print("Exactitud media: {:.2f}".format(scores.mean()))

# Cargar y procesar las nuevas imágenes
for image_path in new_images:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    keypoints, _ = orb.detectAndCompute(gray, None)

    if len(keypoints) > 0:
        # Predecir la etiqueta utilizando el clasificador SVM entrenado
        predicted_label = svm_classifier.predict([[keypoint.pt[0], keypoint.pt[1], keypoint.size, keypoint.angle, keypoint.response, keypoint.octave, keypoint.class_id] for keypoint in keypoints])

        # Mostrar el resultado de la predicción
        print(f"Etiqueta predicha para la imagen {image_path}: {predicted_label}")
    else:
        print(f"No se detectaron puntos clave en la imagen {image_path}.")
