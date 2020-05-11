#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;


Mat src, src_gray;
Mat dst, detected_edges;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;




void AfficherIMageGris(Mat img ,Mat &imgGrey)
{
   
    cvtColor(img, imgGrey, COLOR_BGR2GRAY, 0);
   // namedWindow("Grey", WINDOW_NORMAL);
    //imshow("Grey", imgGrey);
    
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


void ImageLaplacien(Mat img, Mat imgLaplace, int kernel_size, int scale , int delta, int ddepth)
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
void ImageSobel(Mat img, Mat imgSobel ,int kernel_size, int scale, int delta, int ddepth)
{
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
  
    /// Gradient X
    Sobel(img, grad_x,ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    /// Gradient Y
    Sobel(img, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, imgSobel);
    namedWindow("ImageSobel", WINDOW_NORMAL);
    imshow("ImageSobel",imgSobel);
}
void ImageBinaireAdpMoy(Mat img ,int thresh ,Mat &imgBinaryA)
{
    adaptiveThreshold(img, imgBinaryA, thresh, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 11, 7);
    namedWindow("ABinaryMoy", WINDOW_NORMAL);
    imshow("ABinaryMoy", imgBinaryA);
}

void FiltreMedian(Mat img,Mat &imgMedian,int ksize)
{
    medianBlur(img, imgMedian, ksize);
  
}



void AfficherContours(Mat img, Mat imgBinary, int &nb)
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
        nb++;
    }

    imshow("shapes", img);
    imshow("dst", dst);
}

void FloutageGaussien(Mat img, Mat& Gauss)
{
    GaussianBlur(img, Gauss, Size(11, 11), 0, 0, BORDER_DEFAULT);
   
}



void CannyThreshold( int , void*)
{
    /// Reduce noise with a kernel 9x9
    
    FloutageGaussien(src_gray, detected_edges);

    /// Canny detector
    Canny(detected_edges, detected_edges, lowThreshold,lowThreshold*3, kernel_size);

    /// Using Canny's output as a mask, we display our result
    dst = Scalar::all(0);

    src.copyTo(dst, detected_edges);
    imshow("dst", dst);


}


void AplyHoughTransform(Mat img)
{
    vector<Vec2f> lines;
    HoughLinesP(dst, lines, 1, CV_PI / 180, 80, 30, 10);
    for (size_t i = 0; i < lines.size(); i++)
    {
        line(img, Point(lines[i][0], lines[i][1]),
        Point(lines[i][2], lines[i][3]), Scalar(0, 0, 255), 3, 8);
    }
}


float euclideanDist(Point& p, Point& q) {
    Point diff = p - q;
    return cv::sqrt(diff.x * diff.x + diff.y * diff.y);
}

int main()
{

    // count the number of steps
    int countS = 0;
    //ount the number of Csteps
    int countC = 0;

    // Read the image
    src = imread("C:/Users/aya-k/OneDrive/Bureau/StairsDb/img11.png");
    AfficherImageNormal(src);
   // Trransform to greyscale
    AfficherIMageGris(src, src_gray);

    dst.create(src.size(), src.type());

     // Create a window
    namedWindow("CanyDetection", WINDOW_AUTOSIZE);

    //Apply CanyThreshold

    lowThreshold = 30;
    ///Create a Trackbar for user to enter threshold
    createTrackbar("Min Threshold:", "CanyDetection", &lowThreshold, max_lowThreshold, CannyThreshold);
    CannyThreshold(0,0);

  
    
  
   //Detect the horizontal lines
    vector<Vec2f> lines;
    vector<vector<Point>> possibleSteps;
    HoughLines(detected_edges, lines, 1, CV_PI /90, 150);

    for (size_t i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0];
        float theta = lines[i][1];
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        Point pt1(cvRound(x0 + 1000 * (-b)),
            cvRound(y0 + 1000 * (a)));
        Point pt2(cvRound(x0 - 1000 * (-b)),
            cvRound(y0 - 1000 * (a)));
        if (theta>=1.50 && theta <1.80)
        {
           // cout << "theta= " << theta << endl;
            //line(src, pt1, pt2, Scalar(0, 0, 255), 3, 8);
            vector<Point> pt;
            pt.push_back(pt1);
            pt.push_back(pt2);
            possibleSteps.push_back(pt);
           
        }
    }



    // Sort the vector of line by hight 
    for(size_t i = 0; i < possibleSteps.size();i++)
    {
        for (size_t j = 0; j < possibleSteps.size()-1;j++)
        {
            if (possibleSteps[j][0].y < possibleSteps[j + 1][0].y)
            {
                Point p1; 
                Point p2;
                p1 = possibleSteps[j][0];
                p2 = possibleSteps[j][1];
                possibleSteps[j] = possibleSteps[j + 1];
                possibleSteps[j + 1][0] = p1;
                possibleSteps[j + 1][1] = p2;

            }
        }
    }
    

    vector<Point> fline;
    fline = possibleSteps[1];
 
 




    float dist = euclideanDist(fline[0], possibleSteps[1][0]);
    cout << "dist f = " << dist << endl;
    

    for (size_t j = 0; j < possibleSteps.size();j++)
    {
       
        float dist = euclideanDist(fline[0], possibleSteps[j][0]);
        
        cout << "dist entre "  << fline[0].y <<" et " << possibleSteps[j][0].y <<" = "<< dist<< endl;
       
        {
          if (dist >= 25 )
            {
                line(src, possibleSteps[j][0], possibleSteps[j][1], Scalar(0, 0, 255), 3, 8);
                namedWindow("Detected Lines", WINDOW_NORMAL);
                imshow("Detected Lines", src);
                countS++;
            }
        
        }
       
        fline = possibleSteps[j];

        
    }
   
    /****detect vertical lines *******/

    Mat vertical = detected_edges.clone();
    int vertical_size = vertical.rows / 30;
    Mat verticalStructure = getStructuringElement(MORPH_RECT, Size(1, vertical_size));
    // Apply morphology operations
    erode(vertical, vertical, verticalStructure, Point(-1, -1));
    dilate(vertical, vertical, verticalStructure, Point(-1, -1));
    // Show extracted vertical lines

    imshow("vertical", vertical);
    AfficherContours(src, vertical,countC);



    imshow("Detected Lines", src);

    //imshow("Detected LinesC", dst);
    cout << "nombre de marches = " << countS << endl;
    cout << "nombre de contres marches = " << countC << endl;

    namedWindow("Source", 1);
    imshow("Source", detected_edges);

    

       
       




   

   
  

  
    
    
    waitKey(0);
    destroyAllWindows();
    return 0;
}
