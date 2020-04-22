#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;
int main()
{
  // Mat img = imread("C://lena.jpg.png");
   /* Mat img = imread("C:/Users/aya-k/OneDrive/Bureau/Cours-Tp-Td/Analyse d'images/TpImage/test.png");
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
   */
  
   Mat img = imread("C:/Users/aya-k/OneDrive/Bureau/Cours-Tp-Td/Analyse d'images/TpImage/shapes.jpg");
   namedWindow("imageNormal", WINDOW_NORMAL);
   imshow("imageNormal", img);
   Mat  imgGrey;
   cvtColor(img, imgGrey,COLOR_BGR2GRAY,0);
   namedWindow("imageGrey", WINDOW_NORMAL);
   imshow("imageGrey",imgGrey);
   Mat img_not;
   bitwise_not(img, img_not);
   namedWindow("imageNot", WINDOW_NORMAL);
   imshow("imageNot", img_not);
   Mat imgBinary; 
   int thresh = 220;
   int maxval = 255;
   threshold(imgGrey, imgBinary, thresh, maxval, THRESH_BINARY);
   namedWindow("imgBinary", WINDOW_NORMAL);
   imshow("imgBinary",imgBinary);
   Mat contours;
   findContours(imgBinary,contours,RETR_LIST, CHAIN_APPROX_NONE);


  waitKey();

    

    
    return 0;
}