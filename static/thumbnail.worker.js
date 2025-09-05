// static/thumbnail.worker.js
let video = null;
let canvas = null;
let ctx = null;
let currentVideoSrc = null;

// Hàm để chuẩn bị video và canvas
// Trả về một Promise, resolve khi video sẵn sàng để seek
function setupVideo(src) {
    return new Promise((resolve, reject) => {
        if (currentVideoSrc === src && video && video.readyState >= 2) {
            resolve();
            return;
        }

        currentVideoSrc = src;
        video = new OffscreenCanvas(1, 1).getContext('2d').canvas.ownerDocument.createElement('video');
        video.crossOrigin = "anonymous";
        video.muted = true;
        video.src = src;

        video.addEventListener('loadeddata', () => {
            // Sử dụng OffscreenCanvas để tăng hiệu năng, không cần DOM
            canvas = new OffscreenCanvas(video.videoWidth, video.videoHeight);
            ctx = canvas.getContext('2d');
            resolve();
        }, { once: true });

        video.addEventListener('error', (e) => {
            console.error('Worker: Video load error', e);
            reject(new Error('Failed to load video in worker.'));
        }, { once: true });
    });
}


// Lắng nghe tin nhắn từ luồng chính
self.onmessage = async (event) => {
    const { videoSrc, time } = event.data;

    try {
        await setupVideo(videoSrc);

        video.currentTime = time;

        video.addEventListener('seeked', () => {
            // Vẽ frame video lên canvas
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            // Chuyển canvas thành ImageBitmap để gửi về (hiệu năng cao, zero-copy)
            const imageBitmap = canvas.transferToImageBitmap();
            
            // Gửi kết quả về luồng chính
            self.postMessage({
                time: time,
                bitmap: imageBitmap
            }, [imageBitmap]); // Transferable object

        }, { once: true });

    } catch (error) {
        console.error(`Worker: Failed to generate thumbnail for time ${time}:`, error);
        self.postMessage({ time: time, error: error.message });
    }
};