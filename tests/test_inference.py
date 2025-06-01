from models.inference import predict_from_image

if __name__ == "__main__":
    image_path = "static/spectrogramme.png"  # chemin de test
    prediction = predict_from_image(image_path)
    print("Transcription :", prediction)
