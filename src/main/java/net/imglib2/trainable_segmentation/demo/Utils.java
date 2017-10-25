package net.imglib2.trainable_segmentation.demo;

import io.scif.services.DatasetIOService;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.converter.Converters;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.roi.labeling.ImgLabeling;
import net.imglib2.roi.labeling.LabelRegions;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;
import org.scijava.Context;
import org.scijava.plugin.Parameter;

import java.io.IOException;
import java.net.URL;
import java.util.NoSuchElementException;

public class Utils
{
	@Parameter private DatasetIOService ioService;

	Utils( Context context )
	{
		context.inject( this );
	}

	public LabelRegions< String > setupLabeling( RandomAccessibleInterval< ? extends IntegerType< ? > > img )
	{
		final ImgLabeling< String, IntType > labeling = new ImgLabeling<>( ArrayImgs.ints( Intervals.dimensionsAsLongArray( img ) ) );
		Views.interval( Views.pair( img, labeling ), labeling ).forEach( p -> {
			int value = p.getA().getInteger();
			if ( value != 0 )
				p.getB().add( Integer.toString( value ) );
		} );
		return new LabelRegions<>( labeling );
	}

	public RandomAccessibleInterval< FloatType > loadFloatImage( String path )
	{
		RandomAccessibleInterval< ? extends RealType< ? > > realTypes = loadImagePlusFromResource( path );
		return Converters.convert( realTypes, ( in, out ) -> out.setReal( in.getRealFloat() ), new FloatType() );
	}

	public RandomAccessibleInterval< UnsignedByteType > loadByteImage( String path )
	{
		RandomAccessibleInterval< ? extends RealType< ? > > realTypes = loadImagePlusFromResource( path );
		return Converters.convert( realTypes, ( in, out ) -> out.setReal( in.getRealFloat() ), new UnsignedByteType() );
	}

	public Img< ? extends RealType< ? > > loadImagePlusFromResource( final String path )
	{
		String fullPath = getResourcePath( path );
		try
		{
			return ioService.open( fullPath ).getImgPlus().getImg();
		}
		catch ( IOException e )
		{
			e.printStackTrace();
		}
		return null;
	}

	private String getResourcePath( String path )
	{
		final URL url = Utils.class.getResource( "/" + path );
		if ( url == null )
			throw new NoSuchElementException( "file: " + path );
		if ( "file".equals( url.getProtocol() ) )
			return url.getPath();
		return url.toString();
	}
}
