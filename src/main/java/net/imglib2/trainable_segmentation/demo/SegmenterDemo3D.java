package net.imglib2.trainable_segmentation.demo;

import bdv.util.BdvFunctions;
import bdv.util.BdvOptions;
import bdv.util.BdvStackSource;
import net.imagej.ImageJ;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.roi.labeling.ImgLabeling;
import net.imglib2.roi.labeling.LabelRegions;
import net.imglib2.trainable_segmention.classification.Segmenter;
import net.imglib2.trainable_segmention.classification.Trainer;
import net.imglib2.trainable_segmention.pixel_feature.filter.GroupedFeatures;
import net.imglib2.trainable_segmention.pixel_feature.filter.SingleFeatures;
import net.imglib2.trainable_segmention.pixel_feature.settings.FeatureSettings;
import net.imglib2.trainable_segmention.pixel_feature.settings.GlobalSettings;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;
import org.scijava.Context;

import java.util.Arrays;

public class SegmenterDemo3D
{

	private final Utils utils = new Utils( new Context(  ) );

	private final RandomAccessibleInterval< FloatType > image = utils.loadFloatImage( "volvox.tif" );

	private final RandomAccessibleInterval< UnsignedByteType > labelIndices = utils.loadByteImage( "volvoxLabeling2.tif" );

	private final LabelRegions< String > labeling = setupLabeling( labelIndices );

	public static void main( String... args )
	{
		new SegmenterDemo3D().demo();
	}

	private void demo()
	{
		ImageJ imagej = new ImageJ();

		GlobalSettings globals = new GlobalSettings(
				GlobalSettings.ImageType.GRAY_SCALE,
				Arrays.asList( 1.0, 2.0, 4.0, 8.0, 16.0 ),
				3.0
		);

		FeatureSettings settings = new FeatureSettings(
				globals,
				SingleFeatures.identity(),
				GroupedFeatures.gauss()//,
				//GroupedFeatures.hessian3D(false),
				//GroupedFeatures.lipschitz(50 ),
				//GroupedFeatures.differenceOfGaussians(),
				//GroupedFeatures.gradient(),
				//GroupedFeatures.max(),
				//GroupedFeatures.min(),
				//GroupedFeatures.mean(),
				//GroupedFeatures.variance(),
				//GroupedFeatures.median()
		);

		Segmenter segmenter = Trainer.train( imagej.op(), image, labeling, settings );
		Img< ByteType > result = segmenter.segment( image );

		BdvStackSource< FloatType > bdv = BdvFunctions.show( image, "image" );
		BdvFunctions.show( labelIndices,"labeling", BdvOptions.options().addTo( bdv ) );
		BdvFunctions.show( result,"segmentation", BdvOptions.options().addTo( bdv ) );
	}

	// -- Helper methods --

	private static LabelRegions< String > setupLabeling( RandomAccessibleInterval< ? extends IntegerType< ? > > img )
	{
		final ImgLabeling< String, IntType > labeling = new ImgLabeling<>( ArrayImgs.ints( Intervals.dimensionsAsLongArray(img)) );
		Views.interval( Views.pair( img, labeling ), labeling ).forEach( p -> {
			int value = p.getA().getInteger();
			if ( value != 0 )
				p.getB().add( Integer.toString( value ) );
		} );
		return new LabelRegions<>( labeling );
	}

}
