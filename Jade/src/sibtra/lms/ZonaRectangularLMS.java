/**
 * 
 */
package sibtra.lms;

import jade.util.leap.Serializable;


/**
 * @author alberto
 *
 */
public class ZonaRectangularLMS extends ZonaLMS implements Serializable {
	

	/**
	 * 
	 */
	private static final long serialVersionUID = -5520087459532534106L;

	/**
	 * Constructor por defecto  ¿para que sea serializable?
	 */
	public ZonaRectangularLMS() {
		
	}
	
	/**
	 * @param rangoAngular
	 * @param resAngular
	 * @param enMilimetros
	 * @param conjunto1
	 * @param queZona
	 */
	public ZonaRectangularLMS(short rangoAngular, short resAngular,
			boolean enMilimetros, boolean conjunto1, byte queZona) {
		super(rangoAngular, resAngular, enMilimetros, conjunto1, queZona);
	}

	/**
	 * Constructor que define todos los parámetros
	 * @param rangoAngular
	 * @param resAngular
	 * @param enMilimetros
	 * @param conjunto1
	 * @param queZona
	 * @param distIzda
	 * @param distDecha
	 * @param distFrente
	 */
	public ZonaRectangularLMS(short rangoAngular, short resAngular,
			boolean enMilimetros, boolean conjunto1, byte queZona
			,short distIzda,short distDecha, short distFrente) {
		super(rangoAngular, resAngular, enMilimetros, conjunto1, queZona);
		distanciaIzda=distIzda;
		distanciaDecha=distDecha;
		distanciaFrente=distFrente;
	}

	public short distanciaIzda;
	
	public short distanciaDecha;
	
	public short distanciaFrente;

	/**
	 * Trata de interpretar el mensaje C5 para obtener zona rectangular
	 * @param men
	 * @return null si el mensaje no encaja
	 */
	public static ZonaRectangularLMS mensajeC5AZona(byte[] men) {
		//Primeras comprobaciones de tipo y tamaño
		if(men==null || men.length!=20 || men[0]!=((byte)0xc5) )
			return null;
		boolean conjunto1=(men[1]==1);
		boolean enMilimetros=(((int)men[2]&0xc0)>>6)==1;
		byte zona;
		if((men[2]&0xF)==0)
			zona=ZONA_A;
		else if ((men[2]&0x0f)==6)
			zona=ZONA_B;
		else if ((men[2]&0x0f)==0x0C)
			zona=ZONA_C;
		else
			return null;
		ZonaRectangularLMS nz=new ZonaRectangularLMS((short)UtilMensajes.men2Word(men, 3)
				,(short)UtilMensajes.men2Word(men, 5)
				,enMilimetros
				,conjunto1
				,zona);
		nz.distanciaIzda=(short)UtilMensajes.men2Word(men, 13);
		nz.distanciaDecha=(short)UtilMensajes.men2Word(men, 15);
		nz.distanciaFrente=(short)UtilMensajes.men2Word(men, 17);
		return nz;
	}
	
	/**
	 * Devuelve mensaje 40 de definición de zonas correspondiente a esta zona.
	 * @return mensaje 40 completo
	 */
	public byte[] aMensaje40() {
		byte[] men=new byte[20];
		men[7]=0;  //es rectangular
		UtilMensajes.word2Men(distanciaIzda, men, 14);
		UtilMensajes.word2Men(distanciaDecha, men, 16);
		UtilMensajes.word2Men(distanciaFrente, men, 18);
		//Terminamos de rellenar el mensaje.
		return rellenaMensaje40(men);
	}
}
