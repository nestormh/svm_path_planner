/**
 * 
 */
package sibtra.imu;

/**
 * Clase con una serie de utilidades para el manejo de mensajes de la IMU
 * 
 * @author alberto
 *
 */
public class UtilMensajesIMU  {



	/**
	 * Devuelve String con la representación en hexadecimal de los byte de buf.
	 * 
	 * @param buf buffer
	 * @param inic posición inicial del buffer
	 * @param len numero (máximo) de bytes a tratar
	 * @return string en hexa
	 */
	public static String hexaString(byte[] buf, int inic, int len) {
		String resultado=new String();
		for(int i=inic; i<buf.length & i<(inic+len); i++)
			resultado+=Integer.toString((int)buf[i]&0xff, 16).toUpperCase()+" ";
		return resultado.trim();
	}

	/**
	 * Devuelve String con la representación en hexadecimal de los byte de buf.
	 * 
	 * @param buf buffer
	 */ 
	public final static String hexaString(byte[] buf) {
		return hexaString(buf,0,buf.length);
	}
	
	/**
	 * Maneja un mensaje y devuelve las distancias verticales a las 3 zonas en milímetros.
	 * @param mensaje Mensaje a interpretar.
	 * @return las distancias de las tres zonas en milímetros
	 */
	public static double[] mensajeADistancia(byte[] mensaje) {
		// Nos aseguramos que tiene al menos el tipo y que es 0xb0
		if(mensaje==null || mensaje.length<1 || mensaje[0]!=(byte)0xb0)
			return null;
			
		//Cosas particulares de cada tipo
		int campoNumDatos=0;

		if(mensaje.length<3) return null;
		campoNumDatos=men2Word(mensaje,1);
		
		int numDatos=campoNumDatos&(0x1ff); //numero del datos del barrido: bits del 0 a 9
		//int codParcial=campoNumDatos&(0x03<<11)>>11; //que barrido parcial es
		//boolean esParcial=(campoNumDatos&(1<<13))==1;
		int mDist=(campoNumDatos&(0x03<<14))>>14; //codigo para las unidades de medida (cm , mm otro)
				
		if(mDist>1)
			return null; //codigo de distancia reservado
		
		if(numDatos!=3 || mensaje.length<(3+2*3) )
			return null; //no tiene el tamaño esperado
		
		double[] distancias=new double[3];
		
		distancias[0]=(double)UtilMensajesIMU.men2Word(mensaje, 3)*((mDist==0)?10.0:1.0);
		distancias[1]=(double)UtilMensajesIMU.men2Word(mensaje, 5)*((mDist==0)?10.0:1.0);
		distancias[2]=(double)UtilMensajesIMU.men2Word(mensaje, 7)*((mDist==0)?10.0:1.0);
		
		return distancias;
	}

	/**
	 * Devuelve en int un WORD.
	 * WORD como definido en manual LMS: 2 bytes ordenados en big endian.
	 *  
	 * 
	 * @param buf donde están los bytes
	 * @param i posición dentro de buf
	 * @return WORD convertido en int
	 */
	public static int men2Word(byte[] buf, int i) {
		return ((int)buf[i]&0xff) + (((int)buf[i+1]&0xff)<<8);
	}

	

	/**
	 * Mira si el CRC es correcto
	 * @param buf donde se encuentra el mensaje
	 * @param Len se consideran lon len primeros bytes.
	 * @return true si el CRC es correcto
	 */
	public static boolean correctoCRC(byte[] buf,int Len) {
		//comprobamos Check sum
		int sum=0;
		for(int i=1;i<Len;i++) {
			sum+=((int)buf[i])&0xff;
			sum&=0xff;
		}
		return (sum==0);
	}

	/** Coloca byte de checksum al final del mensaje pasado
	 * @param men
	 */
	public static void fijaCRC(byte[] men) {
		int cs=0;
		for(int i=1; i<men.length-1; i++) {
		    cs+=((int)men[i])&0xff;
		    cs&=0xff;
		}
		men[men.length-1]=(byte)(-cs&0xff);
	}

	/**
	 * Coloca los 2 bytes del un Word en big endian en la posicion i del buf
	 * @param w  word a poner
	 * @param buf	buffer donde colocarlo
	 * @param i posición dentro del buffer
	 */
	public static void word2Men(short w, byte[] buf, int i) {
		buf[i]=(byte)(w%256);  //menos significativo primero
		buf[i+1]=(byte)(w/256);  //más significativo después
	}
	
	/**
	 * Compara subrangos de dos mensajes
	 * @param m1 primer mensajes
	 * @param ini1 ínidice inicial del primer mensaje
	 * @param m2 segundo mensaje
	 * @param ini2 índice inicial de segundo mensaje
	 * @param largo longitud del subrango a comparar
	 * @return si son igules los subrangos
	 */
	public static boolean igualesSubMensajes(byte[] m1, int ini1, byte[] m2, int ini2, int largo) {
		boolean iguales=(m1.length>=(ini1+largo)) && (m2.length>=(ini2+largo));
		
		for(int i=0; i<largo && iguales; i++, ini1++, ini2++)
			iguales=(m1[ini1]==m2[ini2]);
		return iguales;
	}
	
	/**
	 * Nos dice si un mensaje es la respuesta de confirmación de otro.
	 * Varios comando 3B, 77, etc. responden con un mensaje que:
	 * <ul>
	 *   <li> Tiene 2 bytes más: mensaje enviado + confirmación + estado
	 *   <li> confirmación vale 1
	 *   <li> respuesta de codigo mensaje & 0x80 (eso pasa con todas las respuestas ;-)
	 * </ul>
	 *   
	 * @param mResp mensaje de respuesta
	 * @param mComando mensaje de comando enviado
	 * @return true si se cumplen las condiciones
	 */
	public static boolean esConfirmacion(byte[] mResp, byte[] mComando) {
		return ( mResp.length==(mComando.length+2) ) //tiene la longitud adecuada
			&& ( mResp[0]==(mComando[0]|(byte)0x80) ) //es del tipo adecudo
			&& ( mResp[1]==1) //Es de aceptación
			&& ( igualesSubMensajes(mComando, 1, mResp, 2, mComando.length-1) ) //resto del mensaje es igual
			;
	}
	
//	public static float men2float(byte[] buf,int i) {
//		float resp;
//		
//		return resp;
//	}
}
