# CVF_SID_Net
Denoising

I downloaded the official weights, but the PSNR=34.05, not 34.67 in paper.(https://github.com/Reyhanehne/CVF-SID_PyTorch/issues/8)
I implemented CVF-SID Net in mmagic, PSNR=34.68

Actually， due to the limitation of my GPU (RTX4060 mobile), I have to reduce batch_size from 64 to 24, num_work from 12 to 8,
and have to use fp16 to avoid OOM in mmagic. However, the result is better than official weights, which confuses me a lot.
