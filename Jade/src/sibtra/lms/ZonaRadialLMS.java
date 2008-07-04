/**
 * 
 */
package sibtra.lms;

import jade.util.leap.Serializable;


/**
 * @author alberto
 *
 */
public class ZonaRadialLMS extends ZonaLMS implements Serializable {
	
	/**
	 * Constructor para que sea serializable
	 */
	public ZonaRadialLMS() {
		super();
	}
	
	/**
	 * @param rangoAngular
	 * @param resAngular
	 * @param enMilimetros
	 * @param conjunto1
	 * @param queZona
	 */
	public ZonaRadialLMS(short rangoAngular, short resAngular,
			boolean enMilimetros, boolean conjunto1, byte queZona) {
		super(rangoAngular, resAngular, enMilimetros, conjunto1, queZona);
	}

	public ZonaRadialLMS(short rangoAngular, short resAngular,
			boolean enMilimetros, boolean conjunto1, byte queZona, short radio) {
		super(rangoAngular, resAngular, enMilimetros, conjunto1, queZona);
		this.radioZona=radio;
	}

	/**
	 * Radio que define la zona.
	 */
	public short radioZona;

	/**
	 * Trata de interpretar el mensaje C5 para obtener zona rectangular
	 * @param men
	 * @return null si el mensaje no encaja
	 */
	public static ZonaRadialLMS mensajeC5AZona(byte[] men) {
		//Primeras comprobaciones de tipo y tamaño
		if(men==null || men.length!=16 || men[0]!=((byte)0xc5) )
			return null;
		boolean conjunto1=(men[1]==1);
		boolean enMilimetros=(((int)men[2]&0xc0)>>6)==1;
		byte zona;
		if((men[2]&0x0f)==1)
			zona=ZONA_A;
		else if ((men[2]&0x0f)==7)
			zona=ZONA_B;
		else if ((men[2]&0x0f)==0x0d)
			zona=ZONA_C;
		else
			return null;
		ZonaRadialLMS nz=new ZonaRadialLMS((short)UtilMensajes.men2Word(men, 3)
				,(short)UtilMensajes.men2Word(men, 5)
				,enMilimetros
				,conjunto1
				,zona);
		nz.radioZona=(short)UtilMensajes.men2Word(men, 13);
		return nz;
	}
	
	/**
	 * Devuelve mensaje 40 de definición de zonas correspondiente a esta zona.
	 * @return mensaje 40 completo
	 */
	public byte[] aMensaje40() {
		byte[] men=new byte[16];
		men[7]=1;  //es radial
		UtilMensajes.word2Men(radioZona, men, 14);
		//Terminamos de rellenar el mensaje.
		return rellenaMensaje40(men);
	}

}
