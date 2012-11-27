package org.opencv.samples.tutorial2;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;
import org.opencv.imgproc.Imgproc;
import org.opencv.features2d.*;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.util.Log;
import android.view.SurfaceHolder;

class Sample2View extends SampleCvViewBase {
    private Mat mRgba;
    private Mat mGray;
    private Mat mIntermediateMat;
    
    private Mat mLast = null;
    private Mat mLastDes = null;
    private MatOfKeyPoint mLastFeatures = null;
    private List<Double> trend;
    private List<Long> trendT;

    public Sample2View(Context context) {
        super(context);
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        synchronized (this) {
            // initialize Mats before usage
            mGray = new Mat();
            mRgba = new Mat();
            mIntermediateMat = new Mat();
            trend = new ArrayList<Double>();
            trendT = new ArrayList<Long>();
        }
        
        super.surfaceCreated(holder);
    }

    @Override
    protected Bitmap processFrame(VideoCapture capture) {
        switch (Sample2NativeCamera.viewMode) {
        case Sample2NativeCamera.VIEW_MODE_GRAY:
            capture.retrieve(mGray, Highgui.CV_CAP_ANDROID_GREY_FRAME);
            Imgproc.cvtColor(mGray, mRgba, Imgproc.COLOR_GRAY2RGBA, 4);
            break;
        case Sample2NativeCamera.VIEW_MODE_RGBA:
            capture.retrieve(mRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
            Core.putText(mRgba, "OpenCV + Android", new Point(10, 100), 3, 2, new Scalar(255, 0, 0, 255), 3);
            break;
        case Sample2NativeCamera.VIEW_MODE_CANNY:
            capture.retrieve(mGray, Highgui.CV_CAP_ANDROID_GREY_FRAME);
            capture.retrieve(mRgba, Highgui.CV_CAP_ANDROID_COLOR_FRAME_RGBA);
            
            
            
            //get the features
            MatOfKeyPoint features = new MatOfKeyPoint();
            FeatureDetector fd = FeatureDetector.create(FeatureDetector.STAR);
            fd.detect(mRgba, features);
            
            
            
            //get the descriptors
            Mat descriptors = new Mat();
            DescriptorExtractor de = DescriptorExtractor.create(DescriptorExtractor.ORB);
            de.compute(mRgba, features, descriptors);

            //first time we have run
            if(mLast == null){
            	mLast = mGray.clone();
            	mLastDes = descriptors;
            	mLastFeatures = features;
            	break;
            }
            
            //match the descriptors to the previous frame
            MatOfDMatch match = new MatOfDMatch();
            DescriptorMatcher dm = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
            dm.match(descriptors, mLastDes, match);
            
            List<KeyPoint> kp = features.toList();
            List<KeyPoint> kp_last = mLastFeatures.toList();
            List<DMatch> all_match = match.toList();
            
            double avgXdiff = 0;
            double avgYdiff = 0;
            for(int i = 0;i < all_match.size();i++){
            	int c_p = all_match.get(i).queryIdx;
            	int l_p = all_match.get(i).trainIdx;
            	double xc = kp.get(c_p).pt.x;
            	double yc = kp.get(c_p).pt.y;
            	double xl = kp_last.get(l_p).pt.x;
            	double yl = kp_last.get(l_p).pt.y;
            	
            	avgXdiff += xc - xl;
            	avgYdiff += yc - yl;
            }
            if(all_match.size() > 0){
            	avgXdiff /= (double)all_match.size();
            	avgYdiff /= (double)all_match.size();
            }
            
            double tot = Math.abs(avgXdiff) + Math.abs(avgYdiff);
            Log.w("Movement", System.nanoTime() + " " + avgXdiff + " " + avgYdiff);
            avgXdiff /= tot;
            avgYdiff /= tot;
            
            trend.add(avgXdiff);
            trendT.add(System.nanoTime());
            
            TrendFinder tf = new TrendFinder();
            double rate = tf.getRespRate(trendT,trend);
            
            //mLast = mGray.clone();
        	
        	
        	//Core.putText(mRgba, "Rate: " + rate, new Point(10, 100), 3, 2, new Scalar(255, 0, 0, 255), 3);
            //if(true)
            	//break;
            
            /*Core.rectangle(mGray, new Point(mGray.width()/4, mGray.height()/2), 
            		new Point(mGray.width()/4 + (avgXdiff*40+1), mGray.height()/2 + avgYdiff*40 + 1), new Scalar(0, 0, 0), Core.FILLED);*/
            
            
           
            /*MatOfKeyPoint features_good = new MatOfKeyPoint();
            MatOfKeyPoint features_good2 = new MatOfKeyPoint();
            List<KeyPoint> kp = features.toList();
            List<KeyPoint> kp_old = mLastFeatures.toList();
            List<KeyPoint> kp_out = new ArrayList<KeyPoint>();
            List<KeyPoint> kp_out2 = new ArrayList<KeyPoint>();
            List<Double> dist = new ArrayList<Double>();
            List<DMatch> all_match = match.toList();
            for(int i = 0;i < all_match.size();i++){
            	DMatch m = all_match.get(i);
            	//min_dist = Math.min(m.distance, min_dist);
            	dist.add((double) m.distance);
            }
            Collections.sort(dist);
            double min_dist = 1e99;
            if(dist.size() > 50)
            	min_dist = dist.get(50);
            MatOfDMatch good_match = new MatOfDMatch();
            List<DMatch> good_tmp = new ArrayList<DMatch>();
            for(int i = 0;i < match.rows();i++){
            	DMatch m = all_match.get(i);
            	if(m.distance <= min_dist)
            		good_tmp.add(m);
            	kp_out.add(kp.get(m.queryIdx));
            	kp_out2.add(kp_old.get(m.trainIdx));
            }
            good_match.fromList(good_tmp);
            features_good.fromList(kp_out);
            features_good2.fromList(kp_out2);*/
            
            Features2d f2d = new Features2d();
            Features2d.drawKeypoints(mGray, features, mIntermediateMat, new Scalar(255, 0, 0, 255), 0);
            //Features2d.drawMatches(mLast, mLastFeatures, mGray, features, match, mRgba, Scalar.all(-1), Scalar.all(-1), new MatOfByte(), Features2d.NOT_DRAW_SINGLE_POINTS);
            //Features2d.drawKeypoints(mRgba, features, mRgba, new Scalar(255, 0, 0, 255), 0);
            Features2d.drawKeypoints(mIntermediateMat, mLastFeatures, mRgba, new Scalar(0, 0, 255, 255), 0);
            //Imgproc.cvtColor(mGray, mRgba, Imgproc.COLOR_GRAY2BGRA, 4);
            mLastDes = descriptors;
        	mLastFeatures = features;
			break;
        }

        Bitmap bmp = Bitmap.createBitmap(mRgba.cols(), mRgba.rows(), Bitmap.Config.ARGB_8888);

        try {
        	Utils.matToBitmap(mRgba, bmp);
            return bmp;
        } catch(Exception e) {
        	Log.e("org.opencv.samples.tutorial2", "Utils.matToBitmap() throws an exception: " + e.getMessage());
            bmp.recycle();
            return null;
        }
    }

    @Override
    public void run() {
        super.run();

        synchronized (this) {
            // Explicitly deallocate Mats
            if (mRgba != null)
                mRgba.release();
            if (mGray != null)
                mGray.release();
            if (mIntermediateMat != null)
                mIntermediateMat.release();

            mRgba = null;
            mGray = null;
            mIntermediateMat = null;
        }
    }
}
