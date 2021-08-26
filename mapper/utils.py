import os
google_drive_paths = {
  "stylegan2-ffhq-config-f.pt": "https://drive.google.com/uc?id=1v8psSmJvXXJbBm_kqLtQv7GeUOxLyv7U",
  "model_ir_se50.pth": "https://drive.google.com/uc?id=1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn",
  "e4e_ffhq_encode.pt": "https://drive.google.com/uc?id=1nR2rnRiK0HKPs-JTMmoHsRs8zE2kdSnQ",
  "shape_predictor_68_face_landmarks.dat": "https://drive.google.com/uc?id=1NBgh-ypy49K6okKsLOUwtqyoEnAXBxoy",
  
  "train_data.pt":"https://drive.google.com/uc?id=1gof8kYc_gDLUT4wQlmUdAtPnQIlCO26q",
  "train_female.pt": "https://drive.google.com/uc?id=1_ZTJa9VmhWaU5xyzyAdK4GHxhnzaEORx",
  "train_male.pt" : "https://drive.google.com/uc?id=1u9r3qcH7qqGHGoolkaR-xEyNCDIz0bCv",

  "test_data.pt" : "https://drive.google.com/uc?id=1j7RIfmrCoisxx3t-r-KC02Qc8barBecr",
  "test_female.pt" : "https://drive.google.com/uc?id=1LYOdh-45aNbGwaj38qQObbBX73wMJXu1",
  "test_male.pt" : "https://drive.google.com/uc?id=11MGereOsRMUo_HQXV8NnFR-L_ECcTXvk",

  "color_sum.pt": "https://drive.google.com/uc?id=1AKYJgRdq76SDQ0xu_VAnGDlm5rmIIjuq",
  "celeb_female_sum.pt" : "https://drive.google.com/uc?id=1ZoFlL6DolbjVAHbCbSY41EDM3Ytzn6PT",
  "celeb_male_sum.pt": "https://drive.google.com/uc?id=1LP5RrPR27Jc_hgFYpjcu2KRbF8yy7OSj",
  "hairstyle_sum.pt": "https://drive.google.com/uc?id=1IIJ78i8s-8uHfRnKt6OVPTnjwFE6Q2CZ",
  "color_cat.pt": "https://drive.google.com/uc?id=1xDRfYG-knfSumbBELWVjscFkEDfQ0UJJ",
  "hairstyle_cat.pt": "https://drive.google.com/uc?id=1aLx7iOhpmmfIeHSnBpscDyk98nLajk5K",
  "color_clip.pt": "https://drive.google.com/uc?id=1s_UufoE8cGaihqDYYdD07CsBUyWpf0CH",
  "hairstyle_clip_cat.pt": "https://drive.google.com/uc?id=1chUIchCEtyU4Mrg0BBjvKAcNKPtABELG",
  "Disney_clip_cat.pt": "https://drive.google.com/uc?id=1OvntzHYCdhonuNsEZAw_qf0wbQ4xp5y2",
    
  "multi_sum.pt": "https://drive.google.com/uc?id=1BY0_uivScHE1_2wcDCEDS0WsLE6LOFiZ"
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
