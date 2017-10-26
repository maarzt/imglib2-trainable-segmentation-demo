package net.imglib2.trainable_segmentation.demo;

import bdv.util.Bdv;
import bdv.util.BdvFunctions;
import bdv.util.BdvOptions;
import bdv.util.BdvStackSource;
import bdv.util.volatiles.SharedQueue;
import bdv.util.volatiles.VolatileViews;
import net.imagej.ImageJ;
import net.imagej.ops.OpService;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.Volatile;
import net.imglib2.cache.img.CellLoader;
import net.imglib2.cache.img.DiskCachedCellImg;
import net.imglib2.cache.img.DiskCachedCellImgFactory;
import net.imglib2.cache.img.DiskCachedCellImgOptions;
import net.imglib2.roi.labeling.LabelRegions;
import net.imglib2.trainable_segmention.classification.Segmenter;
import net.imglib2.trainable_segmention.classification.Trainer;
import net.imglib2.trainable_segmention.pixel_feature.filter.GroupedFeatures;
import net.imglib2.trainable_segmention.pixel_feature.filter.SingleFeatures;
import net.imglib2.trainable_segmention.pixel_feature.settings.FeatureSettings;
import net.imglib2.trainable_segmention.pixel_feature.settings.GlobalSettings;
import net.imglib2.type.numeric.ARGBType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;
import org.scijava.Context;

import java.awt.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class SegmenterDemo3D
{

	private final Utils utils = new Utils( new Context() );

	private final RandomAccessibleInterval< FloatType > image = utils.loadFloatImage( "volvox.tif" );

	private final RandomAccessibleInterval< UnsignedByteType > labelIndices = utils.loadByteImage( "volvoxLabeling2.tif" );

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
		// NB: Using the other features is to slow.
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

		Segmenter segmenter = trainInParallel( imagej.op(), settings, image, labelIndices );
		RandomAccessibleInterval< Volatile< UnsignedByteType > > result = segmentCachedAndLazy( segmenter, image );

		Bdv bdv = showImage( "image", image, Color.red, 200, null );
		showImage( "labeling", labelIndices, Color.blue, 2, bdv );
		showImage( "segmentation", result, Color.green, 2, bdv );
	}

	private Segmenter trainInParallel( OpService ops,
			FeatureSettings settings,
			RandomAccessibleInterval< ? > image,
			RandomAccessibleInterval< UnsignedByteType > labelIndices )
	{
		List<String> classNames = new ArrayList<>( utils.setupLabeling( labelIndices ).getExistingLabels() );
		Segmenter segmenter = new Segmenter( ops, classNames, settings, Trainer.initRandomForest() );
		Trainer trainer = Trainer.of( segmenter );
		trainer.start();
		ParallelUtils.chunkAndRunOperation( labelIndices, new int[]{100, 100, 60}, labelIndicesTile -> {
			LabelRegions< String > labeling = utils.setupLabeling( labelIndicesTile );
			if(!labeling.getExistingLabels().isEmpty())
				trainer.trainLabeledImage( image, labeling );
		} );
		trainer.finish();
		return segmenter;
	}

	private RandomAccessibleInterval< Volatile< UnsignedByteType > > segmentCachedAndLazy( Segmenter segmenter, RandomAccessibleInterval< FloatType > image )
	{
		RandomAccessible< FloatType > extendedImage = Views.extendBorder( image );
		CellLoader< UnsignedByteType > loader = target -> segmenter.segment( target, extendedImage );
		return wrapAsVolatile( image, loader, new int[] { 50, 50, 10 } );
	}

	private RandomAccessibleInterval< Volatile< UnsignedByteType > > wrapAsVolatile( RandomAccessibleInterval< FloatType > image, CellLoader< UnsignedByteType > byteTypeCellLoader, int[] cellDimensions )
	{
		DiskCachedCellImgFactory<UnsignedByteType> factory = new DiskCachedCellImgFactory<>( new DiskCachedCellImgOptions().cellDimensions( cellDimensions ) );
		SharedQueue queue = new SharedQueue(Runtime.getRuntime().availableProcessors());
		DiskCachedCellImg< UnsignedByteType, ? > result = factory.create( Intervals.dimensionsAsLongArray( image ), new UnsignedByteType(),
				byteTypeCellLoader );
		return VolatileViews.wrapAsVolatile( result, queue );
	}

	private < T > Bdv showImage( String text, RandomAccessibleInterval< T > result, Color color, double maxIntensity, Bdv bdv )
	{
		BdvStackSource< T > b = BdvFunctions.show( result, text, BdvOptions.options().addTo( bdv ) );
		b.setColor( new ARGBType( color.getRGB() ) );
		b.setDisplayRange( 0, maxIntensity );
		return b;
	}
}
