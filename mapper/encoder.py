def encoder(options):
  model_path = "e4e_ffhq_encode.pt"
  ensure_checkpoint_exists(model_path)
  ckpt = torch.load(model_path, map_location=device)
  opts = ckpt['opts']
  opts['checkpoint_path'] = model_path
  opts = Namespace(**opts)
  net = pSp(opts)
  net.eval().to(device)
