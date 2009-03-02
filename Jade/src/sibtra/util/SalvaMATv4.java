/**
 * 
 */
package sibtra.util;



import java.io.BufferedOutputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Iterator;
import java.util.List;
import java.util.Vector;

/**
 * Clase para la gestión de ficheros MAT versión 4 que son los que entiende Octave y Matlab (espero ;-)
 * 
 * @author alberto
 *
 */
public class SalvaMATv4 {

	DataOutputStream dos=null;
	
//	public SalvaMATv4() {
//		
//	}
	
	/**
	 * Se le pasa nombre del fichero a abrir con {@link #open(String)}
	 */
	public SalvaMATv4(String nomFich) throws IOException {
		open(nomFich);
	}

	/** Abre fichero con el nombre pasado.
	 * Si Existe fichero previo, lo cierra primero
	 * @param nomFich nombre del fichero
	 * @throws IOException
	 */
	public void open(String nomFich) throws IOException { 
		if (nomFich==null || nomFich.length()==0)
			throw new IllegalArgumentException("Necesario nombre de fichero ");
		if(dos!=null)
			close();
		dos=new DataOutputStream(
				new BufferedOutputStream (
						new FileOutputStream(nomFich)
				)
		);
	}

	/** Cierra el fichero abierto */
	public void close() throws IOException {
		if(dos!=null)
			dos.close();
	}

	private void escribeEntero(int ent ) throws IOException {
		dos.writeByte(ent);
		dos.writeByte(ent>>8);
		dos.writeByte(ent>>16);
		dos.writeByte(ent>>24);
	}

	
	/** Escribe la cabecera invirtiendo en orden da cada entero.
	 * Se comprueba que la matriz tenga nombre.
	 * @param MOPT Tipo, orden, etc. de la matriz
	 * @param mrows nómero de filas
	 * @param ncols número de columnas
	 * @param imagf 1 si tiene parte imaginaria, 0 si es sólo real
	 * @param nombre de la matriz, ha de se !=Null y >0
	 * @throws IOException
	 */
	private void escribeCabecera(int MOPT, int mrows, int ncols, int imagf, String nombre) throws IOException {
		if(nombre==null || nombre.length()==0)
			throw new IllegalArgumentException("Matriz debe tener un nombre");
		//Primero MOPT
		escribeEntero(MOPT);
		//Segundo mrows
		escribeEntero(mrows);
		//Tercero ncols
		escribeEntero(ncols);
		//Cuarto imagf
		escribeEntero(imagf);
		
		//quinto namlen
		int namlen=nombre.length()+1;
		escribeEntero(namlen);
		
		//escribimos nombre
		dos.writeBytes(nombre);
		dos.writeByte(0); //0 al final de la cadena de caracteres


	}

	/**
	 * Escribe matriz BIDIMENSIONAL de dobles al fichero. Si es null no se escribe nada. 
	 * Si no tiene elementos se escribe matriz vacía. 
	 * @param d matriz a escribir. 
	 * @param nombre nombre de la matriz
	 * @throws IOException
	 */
	public void matrizDoubles(double[][] d,String nombre) throws IOException {
		if(d==null) return; //si no hay datos no hacemos nada.
		int mrows=d.length;
		int ncols=(mrows==0?0:d[0].length);
		// M=1   Big Endian
		// O=0
		// P=0  double-precision (64-bit)
		// T=0  Numeric (Full) matrix
		escribeCabecera(1000, mrows, ncols, 0, nombre);
		
		//escribimos parte real POR COLUMNAS
		if(mrows>0 && ncols>0)
			for(int col=0; col<d[0].length; col++)
				for(int fil=0; fil<d.length; fil++)
					dos.writeDouble(d[fil][col]);
	}

	/**
	 * Escribe array de dobles al fichero como vector columna. Si es null no se escribe nada. 
	 * Si no tiene elementos se escribe matriz vacía. 
	 * @param d lista a escribir
	 * @param nombre nombre de la matriz
	 * @throws IOException
	 */
	public void vectorDoubles(double[] d,String nombre) throws IOException {
		if(d==null) return; //si no hay datos no hacemos nada.
		int mrows=d.length;
		int ncols=1;
		// M=1   Big Endian
		// O=0
		// P=0  double-precision (64-bit)
		// T=0  Numeric (Full) matrix
		escribeCabecera(1000, mrows, ncols, 0, nombre);
		
		//escribimos parte real POR COLUMNAS
		if(mrows>0)
			for(int fil=0; fil<d.length; fil++)
				dos.writeDouble(d[fil]);
	}

	/**
	 * Escribe una lista de dobles al fichero como vector columna. Si es null no se escribe nada.
	 *  Si no tiene elementos se escribe matriz vacía. 
	 * @param d matriz a escribir
	 * @param nombre nombre de la matriz
	 * @throws IOException
	 */
	public void vectorDoubles(List<Double> d,String nombre) throws IOException {
		if(d==null) return; //si no hay datos no hacemos nada.
		int mrows=d.size();
		int ncols=1;
		// M=1   Big Endian
		// O=0
		// P=0  double-precision (64-bit)
		// T=0  Numeric (Full) matrix
		escribeCabecera(1000, mrows, ncols, 0, nombre);
		
		//escribimos parte real
		if(mrows>0)
			for(Iterator<Double> iele=d.iterator(); iele.hasNext(); )
				dos.writeDouble(iele.next());
	}

	/**
	 * Escribe una lista de Longs al fichero como vector columna de enteros (32-bits) con signo. 
	 *  Si es null no se escribe nada.
	 *  Si no tiene elementos se escribe matriz vacía. 
	 * @param d matriz a escribir
	 * @param nombre nombre de la matriz
	 * @throws IOException
	 */
	public void vectorLongs(List<Long> d,String nombre) throws IOException {
		if(d==null) return; //si no hay datos no hacemos nada.
		int mrows=d.size();
		int ncols=1;
		//MOPT
		// M=1   Big Endian, Ponerlo a 0 no afecta a los enteros, hay que invertir al escribir
		// O=0
		// P=0  double-precision (64-bit)
		// T=0  Numeric (Full) matrix
		escribeCabecera(1020, mrows, ncols, 0, nombre);
		
		//escribimos parte real
		if(mrows>0)
			for(Iterator<Long> iele=d.iterator(); iele.hasNext(); )
				escribeEntero(iele.next().intValue());
//		        dos.writeInt(iele.next().intValue());
	}

	/**
	 * Escribe una lista de Integers al fichero como vector columna de enteros (32-bits) con signo. 
	 *  Si es null no se escribe nada.
	 *  Si no tiene elementos se escribe matriz vacía. 
	 * @param d matriz a escribir
	 * @param nombre nombre de la matriz
	 * @throws IOException
	 */
	public void vectorIntegers(List<Integer> d,String nombre) throws IOException {
		if(d==null) return; //si no hay datos no hacemos nada.
		int mrows=d.size();
		int ncols=1;
		//MOPT
		// M=1   Big Endian, Ponerlo a 0 no afecta a los enteros, hay que invertir al escribir
		// O=0
		// P=2  double-precision (64-bit)
		// T=0  Numeric (Full) matrix
		escribeCabecera(1020, mrows, ncols, 0, nombre);
		
		//escribimos parte real
		if(mrows>0)
			for(Iterator<Integer> iele=d.iterator(); iele.hasNext(); )
				escribeEntero(iele.next());
//				dos.writeInt(iele.next());
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		String nomFich="PruebaV4.mat";
		
		double[][] m1={
				{12,1.5,10e-4,-45}
				,{100, 2000, 30000, 400000}
		};
		
		double[] m2={1.0, 3.33 ,5.55 ,7.77};

		Vector<Double> vd=new Vector<Double>();
		vd.add(10.0);
		vd.add(12.3);
		vd.add(1e-5);
		
		Vector<Long> vl= new Vector<Long>();
		vl.add(10l);
		vl.add(2000000l);
		vl.add(20l);
		
		Vector<Integer> vi= new Vector<Integer>();
		for(int i=-5; i<=10; i++)
			vi.add(i);
		
		SalvaMATv4 smv4;
		
		try {
			smv4=new SalvaMATv4(nomFich);
			smv4.matrizDoubles(m1, "m1");
			smv4.vectorDoubles(m2, "m2");
			smv4.vectorDoubles(vd, "vd");
			smv4.vectorLongs(vl, "vl");
			smv4.vectorIntegers(vi, "vi");
			//cerramos fichero
			smv4.close();
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}

}
