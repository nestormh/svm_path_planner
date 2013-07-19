/**
 * 
 */
package sibtra.lms;

import java.io.Serializable;

/**
 * Clase padre de todas los tipos de zona LMS posibles.
 * @author alberto
 * 
 */
public abstract class ZonaLMS extends ParametrosLMS implements Serializable {

	/**
	 * Si pertenece al conjunto 1 (true) o al conjunto 2 (false)
	 */
	public boolean  conjunto1;
	

	public static byte ZONA_A=0;
	public static byte ZONA_B=1;
	public static byte ZONA_C=2;
	
	/**
	 * A que zona corresponde
	 */
	private byte queZona=ZONA_A;

	
	/**
	 * Constructor por defecto ¿para que sea serializable?
	 */
	public ZonaLMS() {
		
	}
	
	/** Constructor con todos los parámetros
	 * @param rangoAngular
	 * @param resAngular
	 * @param enMilimetros
	 * @param conjunto1
	 * @param queZona
	 */
	public ZonaLMS(short rangoAngular, short resAngular, boolean enMilimetros,
			boolean conjunto1, byte queZona) {
		super(rangoAngular,resAngular,enMilimetros);
		this.conjunto1=conjunto1;
		setQueZona(queZona);
	}

	/**
	 * @return the queZona
	 */
	public byte getQueZona() {
		return queZona;
	}

	/**
	 * Se comprueba que sea uno de los valores válidos
	 * @param queZona the queZona to set
	 * @return true si valor era válido false si no se hizo el cambio
	 */
	public boolean setQueZona(byte queZona) {
		if (queZona==ZONA_A || queZona==ZONA_B || queZona==ZONA_C) {
			this.queZona = queZona;
			return true;
		}
		return false;
	}
		
	public byte[] rellenaMensaje40(byte[] men) {
		if(men==null || men.length<14)
			return null;
		men[0]=(byte)0x40;
		men[1]=(byte)(conjunto1?1:2);
		men[2]=(byte)(getQueZona()&(isEnMilimetros()?0x40:0));  //zonas vale 0, 1 ó 2
		UtilMensajes.word2Men(getRangoAngular(), men, 3);
		UtilMensajes.word2Men(getResAngular(), men, 5);
		men[8]=0;
		men[9]=0;
		men[10]=0;
		men[11]=0;
		men[12]=0;
		men[13]=0;
		return men;
	}

	public static ZonaLMS MensajeC5AZona(byte[] men) {
		//Primeras comprobaciones de tipo y tamaño
		if(men==null || men.length<3 || men[0]!=((byte)0xc5) )
			return null;
		//elegimos quien hará la conversión
		int decide=(men[2]&0x3f)%6;
		if( decide== 0)
			return ZonaRectangularLMS.mensajeC5AZona(men);
		if(decide == 1)
			return ZonaRadialLMS.mensajeC5AZona(men);
		if(decide == 2 || decide == 3)  //Zonas taught-in se convierten a segmentadas
			return ZonaSegmentadaLMS.MensajeC5AZona(men);
		return null;
	}

	/**
	 * Devuelve si pertenece al conjunto 1 ó 2
	 * @return true si es del 1, false si es del 2
	 */
	public boolean isConjunto1() {
		return conjunto1;
	}
}
