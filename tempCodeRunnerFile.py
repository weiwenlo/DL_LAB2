                print("CUDA available:", torch.cuda.is_available())
                print("Images device:", images.device)
                print("Model param device:", next(model.parameters()).device)
                print("Outputs device:", outputs.device)
