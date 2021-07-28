
google_drive_paths = {
  
  "pink_hair.pt" : "https://drive.google.com/file/d/19aer_dJNRGEUvc3_vurCTQO4719grV1h",
  "Elsa.pt" : "https://drive.google.com/file/d/1YDyYL8katr6yvY_EDU14EOpE7Ay3i-83"
  
  "afro.pt" : "https://drive.google.com/uc?id=1i5vAqo4z0I-Yon3FNft_YZOq7ClWayQJ",
  "angry.pt" : "https://drive.google.com/uc?id=1g82HEH0jFDrcbCtn3M22gesWKfzWV_ma",
  "beyonce.pt" : "https://drive.google.com/uc?id=1KJTc-h02LXs4zqCyo7pzCp0iWeO6T9fz",
  "bobcut.pt" : "https://drive.google.com/uc?id=1IvyqjZzKS-vNdq_OhwapAcwrxgLAY8UF",
  "bowlcut.pt" : "https://drive.google.com/uc?id=1xwdxI2YCewSt05dEHgkpmmzoauPjEnnZ",
  "curly_hair.pt" : "https://drive.google.com/uc?id=1xZ7fFB12Ci6rUbUfaHPpo44xUFzpWQ6M",
  "depp.pt" : "https://drive.google.com/uc?id=1FPiJkvFPG_y-bFanxLLP91wUKuy-l3IV",
  "hilary_clinton.pt" : "https://drive.google.com/uc?id=1X7U2zj2lt0KFifIsTfOOzVZXqYyCWVll",
  "mohawk.pt" : "https://drive.google.com/uc?id=1oMMPc8iQZ7dhyWavZ7VNWLwzf9aX4C09",
  "purple_hair.pt" : "https://drive.google.com/uc?id=14H0CGXWxePrrKIYmZnDD2Ccs65EEww75",
  "surprised.pt" : "https://drive.google.com/uc?id=1F-mPrhO-UeWrV1QYMZck63R43aLtPChI",
  "taylor_swift.pt" : "https://drive.google.com/uc?id=10jHuHsKKJxuf3N0vgQbX_SMEQgFHDrZa",
  "trump.pt" : "https://drive.google.com/uc?id=14v8D0uzy4tOyfBU3ca9T0AzTt3v-dNyh",
  "zuckerberg.pt" : "https://drive.google.com/uc?id=1NjDcMUL8G-pO3i_9N6EPpQNXeMc3Ar1r"

}

def ensure_checkpoint_exists(model_weights_filename):
    if not os.path.isfile(model_weights_filename) and (
        model_weights_filename in google_drive_paths
    ):
        gdrive_url = google_drive_paths[model_weights_filename]
        try:
            from gdown import download as drive_download

            drive_download(gdrive_url, model_weights_filename, quiet=False)
        except ModuleNotFoundError:
            print(
                "gdown module not found.",
                "pip3 install gdown or, manually download the checkpoint file:",
                gdrive_url
            )

    if not os.path.isfile(model_weights_filename) and (
        model_weights_filename not in google_drive_paths
    ):
        print(
            model_weights_filename,
            " not found, you may need to manually download the model weights."
        )
