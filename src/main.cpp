#include "yolov8_seg.h"
#include <chrono>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

std::unique_ptr<Yolo> yolov8Seg(new Yolo());

// ������д�С�������ڴ�����������
const int QUEUE_MAX_SIZE = 10;
std::queue<cv::Mat> frameQueue;
std::mutex mtx;
std::condition_variable cv_frame;
bool stopProcessing = false;

// ��ȡ��Ƶ֡�ĺ���
void readFrames(cv::VideoCapture& cap) {
    while (true) {
        std::unique_lock<std::mutex> lock(mtx);
        cv_frame.wait(lock, [] { return frameQueue.size() < QUEUE_MAX_SIZE || stopProcessing; });

        if (stopProcessing) {
            break;
        }

        cv::Mat frame;
        cap >> frame;

        if (frame.empty()) {
            stopProcessing = true;
            cv_frame.notify_all();
            break;
        }

        frameQueue.push(frame);
        cv_frame.notify_all();
    }
}

// ������Ƶ֡�ĺ���
void processFrames() {
    while (true) {
        std::unique_lock<std::mutex> lock(mtx);
        cv_frame.wait(lock, [] { return !frameQueue.empty() || stopProcessing; });

        if (stopProcessing && frameQueue.empty()) {
            break;
        }

        cv::Mat frame = frameQueue.front();
        frameQueue.pop();
        lock.unlock();
        cv_frame.notify_all();

        std::vector<Object> objects;

        // ��¼��ʼʱ��
        auto start = std::chrono::high_resolution_clock::now();

        // ���� detect ����
        objects.clear();
        yolov8Seg->detect(frame, objects);

        if (objects.size() > 0) {
            yolov8Seg->draw(frame, objects);
        }
        else {
            yolov8Seg->draw_unsupported(frame);
        }

        yolov8Seg->draw_fps(frame);

        // ��¼����ʱ��
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;

        // �������ʱ��
        std::cout << "Processing time per frame: " << elapsed.count() << " ms" << std::endl;

        // ��ʾ��ǰ֡
        cv::imshow("YOLOv8 Segmentation - Video", frame);

        // ���� 'q' ���˳�
        if (cv::waitKey(1) == 'q') {
            stopProcessing = true;
            cv_frame.notify_all();
            break;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <mode> <path>" << std::endl;
        std::cerr << "Mode: single - Process a single image" << std::endl;
        std::cerr << "Folder - Process all images in a folder" << std::endl;
        return -1;
    }

    std::string mode = argv[1];
    std::string path = argv[2];

    if (mode == "image") {
        cv::Mat image = cv::imread(path);
        if (image.empty()) {
            std::cerr << "Could not open or find the image at " << path << std::endl;
            return -1;
        }

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<Object> objects;
        yolov8Seg->detect(image, objects);

        if (!objects.empty()) {
            yolov8Seg->draw(image, objects);
        }
        else {
            yolov8Seg->draw_unsupported(image);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cout << "Processing time: " << elapsed.count() << " ms" << std::endl;

        cv::imshow("YOLOv8 Segmentation - Image", image);
        cv::waitKey(0);
    }
    else if (mode == "video") {
        cv::VideoCapture cap;
        if (path == "0") {
            cap.open(0);  // ��Ĭ������ͷ
        }
        else {
            cap.open(path);  // ����Ƶ�ļ�
        }

        if (!cap.isOpened()) {
            std::cerr << "Could not open video: " << path << std::endl;
            return -1;
        }

        // ������ȡ�ʹ���֡���߳�
        std::thread readerThread(readFrames, std::ref(cap));
        std::thread processorThread(processFrames);

        // �ȴ��߳̽���
        readerThread.join();
        processorThread.join();

        cap.release();
        cv::destroyAllWindows();
    }
    else {
        std::cerr << "Invalid mode. Use 'single' or 'folder'." << std::endl;
        return -1;
    }

    std::cout << "Processing complete" << std::endl;
    return 0;
}
