#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void AfficherIMageGris(Mat img ,Mat &imgGrey)
{
   
    cvtColor(img, imgGrey, COLOR_BGR2GRAY, 0);
    namedWindow("Grey", WINDOW_NORMAL);
    imshow("Grey", imgGrey);
    
}

void AfficherImageBinaire(Mat img , int thresh, int maxval ,Mat &imgBinary)
{
    threshold(img, imgBinary, thresh, maxval, THRESH_BINARY);
    namedWindow("imgBinary", WINDOW_NORMAL);
    imshow("imgBinary", imgBinary);
   
}

void AfficherImageNegative(Mat img)
{
    Mat img_not;
    bitwise_not(img, img_not);
    namedWindow("imageNot", WINDOW_NORMAL);
    imshow("imageNot", img_not);
}

void AfficherImageNormal(Mat img)
{
    namedWindow("Normal", WINDOW_NORMAL);
    imshow("Normal", img);

}
void FloutageGaussien(Mat img, Mat Gauss)
{
    GaussianBlur(img, Gauss, Size(3, 3), 0, 0, BORDER_DEFAULT);
    namedWindow("ImageGauss", WINDOW_NORMAL);
    imshow("ImageGauss", Gauss);
}

void Laplacien(Mat img, Mat imgLaplace, int kernel_size, int scale , int delta, int ddepth)
{
   /* int kernel_size = 5;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
    */
    Mat dst;
    Laplacian(img, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(dst, imgLaplace);
    namedWindow("ImageLaplace", WINDOW_NORMAL);
    imshow("ImageLaplace", imgLaplace);

}
void Sobel(Mat img, int kernel_size, int scale, int delta, int ddepth)
{
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    Mat grad;
    /// Gradient X
    Sobel(img, grad_x,ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    /// Gradient Y
    Sobel(img, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
    namedWindow("ImageSobel", WINDOW_NORMAL);
    imshow("ImageSobel", grad);
}
void ImageBinaireAdpMoy(Mat img ,int thresh ,Mat imgBinaryA)
{
    adaptiveThreshold(img, imgBinaryA, thresh, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 11, 7);
    namedWindow("ABinaryMoy", WINDOW_NORMAL);
    imshow("ABinaryMoy", imgBinaryA);
}

void tp1()
{
   
    Mat img = imread("C:/Users/aya-k/OneDrive/Bureau/Cours-Tp-Td/Analyse d'images/TpImage/test.png");
    if (!img.data)
    {
        cout << "img null" << endl;
        exit(-1);

    }
    namedWindow("imageNormal", WINDOW_NORMAL);
    imshow("imageNormal", img);
    

    int thresh = 145;
    int maxval = 255;
    Mat thresholdedImage;

    threshold(img, thresholdedImage, thresh, maxval, THRESH_BINARY);
    namedWindow("imageThresholded", WINDOW_NORMAL);
    imshow("imageThresholded", thresholdedImage);
}

void tp2()
{
    Mat imgGrey;
    Mat imgBinary;
    Mat img = imread("C:/Users/aya-k/OneDrive/Bureau/StairsDb/shapes.jpg");
    AfficherImageNormal(img);
    AfficherIMageGris(img, imgGrey);
    AfficherImageBinaire(imgGrey, 220, 255, imgBinary);



    Mat dst = Mat::zeros(imgBinary.rows, imgBinary.cols, CV_8UC3);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    vector<vector<Point> > approx;
    vector<Point> points;


    findContours(imgBinary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
    approx.resize(contours.size());

    for (size_t i = 0; i < contours.size(); i++)
    {
        approxPolyDP(contours[i], approx[i], 0.01 * arcLength(contours[i], true), true);
        Scalar color(rand() & 255, rand() & 255, rand() & 255);
        drawContours(img, approx, i, color, 5);
        drawContours(dst, contours, i, color, 2, 8, hierarchy, 0, Point());

        Point x = approx[i][1];
        Point y = approx[i][1];
        if (approx[i].size() == 3)
        {
            cout << "triangle" << endl;
            putText(img, "Triangle", (x, y), FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255));
        }
        else
        {
            if (approx[i].size() == 4)
            {
                Rect rect = boundingRect(approx[i]);
                cout << "x=" << rect.x << "y=" << rect.y << "width =" << rect.width << "height= " << rect.height << endl;
                cout << "x= " << x << "y= " << y << endl;

                float ratio = float(rect.width) / rect.height;

                if (ratio <= 1.05 and ratio >= 0.95)
                {
                    cout << "ratio =" << ratio << "square" << endl;
                    putText(img, "SQUARE", (x, y), FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255));
                }
                else
                {
                    cout << "ratio =" << ratio << "rectangle" << endl;
                    putText(img, "RECTANGLE", (x, y), FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255));
                }




            }
            else
                if (approx[i].size() == 5)
                {
                    cout << "pentagone" << endl;
                    putText(img, "PENTAGONE", (x, y), FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255));
                }
                else
                    if (approx[i].size() == 10)
                    {
                        cout << "star" << endl;
                        putText(img, "STAR", (x, y), FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255));
                    }
                    else
                    {
                        cout << "circle" << endl;
                        putText(img, "CIRCLE", (x, y), FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0));
                    }
        }





        //drawContours(dst, contours, i, color, 2, 8, hierarchy, 0, Point());
    }
    imshow("shapes", img);
    imshow("dst", dst);
}
void AfficherContours(Mat img, Mat imgBinary)
{
    Mat dst = Mat::zeros(imgBinary.rows, imgBinary.cols, CV_8UC3);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    vector<vector<Point> > approx;
    vector<Point> points;


    findContours(imgBinary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
    approx.resize(contours.size());

    for (size_t i = 0; i < contours.size(); i++)
    {
        approxPolyDP(contours[i], approx[i], 0.01 * arcLength(contours[i], true), true);
        Scalar color(rand() & 255, rand() & 255, rand() & 255);
        drawContours(img, approx, i, color, 5);
        drawContours(dst, contours, i, color, 2, 8, hierarchy, 0, Point());
         

    }

    imshow("shapes", img);
    imshow("dst", dst);
}
int main()
{
  
    Mat imgBinaire;
    Mat imgGris;
    Mat img = imread("C:/Users/aya-k/OneDrive/Bureau/StairsDb/img7.jpg");
    AfficherImageNormal(img);
    AfficherIMageGris(img, imgGris);
    AfficherImageBinaire(imgGris, 220, 255, imgBinaire);

    Mat dst = Mat::zeros(img.rows, img.cols, CV_8UC3);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    vector<vector<Point> > approx;
    vector<Point> points;



   findContours(imgBinaire, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
    approx.resize(contours.size());

    for (size_t i = 0; i < contours.size(); i++)
    {
        approxPolyDP(contours[i], approx[i], 0.01 * arcLength(contours[i], true), true);
        Scalar color(rand() & 255, rand() & 255, rand() & 255);
        drawContours(img, approx, i, color, 5);
        drawContours(dst, contours, i, color, 2, 8, hierarchy, 0, Point());

        Point x = approx[i][1];
        Point y = approx[i][1];
        cout << "" << approx[i].size() << endl;

        if (approx[i].size() == 4)
        {
            Rect rect = boundingRect(approx[i]);
           
            float ratio = float(rect.width) / rect.height;

            if (ratio <= 1.05 and ratio >= 0.95)
            {
                cout <<"square" << endl;
               
            }
            else
            {
                cout <<"rectangle" << endl;
              
            }

         
        }
        else
            cout <<"other" << endl;
       


    }

    imshow("shapes", img);
    imshow("dst", dst);
  
  


  

  
    
    
    waitKey(0);
    destroyAllWindows();
    return 0;
}
