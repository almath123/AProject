package org.opencv.samples.tutorial2;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;

import android.util.Log;

public class TrendFinder {
	private static final long OUT_SAMPLE_RATE = 3; //samples per second to pass into fft
	private static final long TIME_RATE = 1000000000; //ticks per second
	private static final double EPS = 0.0000001;
	private static final long MAX_BREATH_LEN = 20; //maximum length of a breath in seconds
	private static final double MIN_BREATH_LEN = 0.75;
	
	double getRespRate(List<Long> tt, List<Double> v){
		if(tt.size() == 0)return 0;
		
		//zero the time series
		List<Long> t = new ArrayList<Long>();
		long offset = tt.get(0).longValue();
		for(int i = 0;i < tt.size();i++){
			t.add(tt.get(i).longValue() - offset);
		}		
		
		//do periodic sampling via two point interpolation
		double timePerSample =  (double)TIME_RATE / (double)OUT_SAMPLE_RATE;
		int lastIdx = 0;
		List<Double> perV = new ArrayList<Double>();
		for(double ct = EPS; ct < t.get(t.size()-1); ct+=timePerSample){
			//Log.w("maxPos", "" + ct + " " + t.get(t.size()-1));
			while(((double)t.get(lastIdx)) <= ct){
				lastIdx+=1;
				if(lastIdx > t.size())break;
			}if(lastIdx > t.size())break;
			
			double diff = (t.get(lastIdx) - t.get(lastIdx-1));
			double lastW =  (diff - (ct - (double)t.get(lastIdx-1)))/diff;
			double curW = 1.0 - lastW;
			perV.add(lastW * v.get(lastIdx-1) + curW * v.get(lastIdx));
		}
		
		
		
		Mat tRep = new Mat();
		Mat fRep = new Mat();
		tRep = Mat.zeros(new Size(1, Core.getOptimalDFTSize((perV.size()+1/2))*2), CvType.CV_64F);
		for(int i = 0;i < perV.size();i++){
			tRep.put(i, 0, perV.get(i), 0);
		}
		/*tRep = Mat.zeros(new Size(1, Core.getOptimalDFTSize((100+1/2))*2), CvType.CV_32FC2);
		for(int i = 1;i <= 100;i++){
			tRep.put(i, 0, i, 0);
		}*/
		Core.dft(tRep, fRep, Core.DFT_COMPLEX_OUTPUT, 0);
		/*for(int i = 0;i < fRep.rows();i++){
			double mag = Math.sqrt(Math.pow(fRep.get(i, 0)[0], 2) + Math.pow(fRep.get(i, 0)[1], 2));
			Log.w("saveData", "" + mag + " R: " + fRep.get(i, 0)[0] + " I: " + fRep.get(i, 0)[1]);
		}
		System.exit(0);*/
		/*for(int i = 0;i < fRep.rows();i++){
			Log.w("maxPos", "DCT: " + );
		}*/
		//if(true)
			//return 0;
		
		double maxV = -1e99;
		int maxPos = 0;
		int fullLen = Core.getOptimalDFTSize((perV.size()+1)/2)*2;
		for(int i = 0;i < fullLen;i++){
			if(1.0 / (((double)i / (double)fullLen) * (double)OUT_SAMPLE_RATE) > MAX_BREATH_LEN)continue;
			if(1.0 / (((double)i / (double)fullLen) * (double)OUT_SAMPLE_RATE) < MIN_BREATH_LEN)break;

			
			double mag = Math.sqrt(Math.pow(fRep.get(i, 0)[0], 2) + Math.pow(fRep.get(i, 0)[1], 2));
			//mag += Math.sqrt(Math.pow(fRep.get(fullLen-i-1, 0)[0], 2) + Math.pow(fRep.get(fullLen-i-1, 0)[1], 2));
			//Log.w("maxPos", "" + mag);
			if(mag > maxV){
				maxV = mag;
				maxPos = i;
			}
		}
		
		/*Log.w("saveData", "" + fRep.rows() + " " + fRep.cols());
		if(perV.size() == 150){
			for(int i = 0;i < fRep.rows();i++){
				double mag = Math.pow(fRep.get(i, 0)[0], 2) + Math.pow(fRep.get(i, 0)[1], 2);
				Log.w("saveData", "" + mag);
			}
		}
		if(perV.size() == 300){
			for(int i = 0;i < fRep.rows();i++){
				double mag = Math.pow(fRep.get(i, 0)[0], 2) + Math.pow(fRep.get(i, 0)[1], 2);
				Log.w("saveData", "" + mag);
			}
		}*/
		
		double freq = ((double)maxPos / ((double) fullLen)) *  (double)OUT_SAMPLE_RATE;
		//if(maxPos > fullLen / 2) maxPos = fullLen - maxPos;
		//Log.w("maxPos", "Raw: " + maxPos + " Rateof: " + (double)maxPos / (double)OUT_SAMPLE_RATE + " gives: " +
			//	60.0 / ((double)maxPos / (double)OUT_SAMPLE_RATE) + " bpm");
		/*Log.w("maxPos", "Raw: " + maxPos + " Freq: " + freq + " gives: " +
			freq * 60.0 + " bpm");*/
		
		return freq * 60;
	}
}
