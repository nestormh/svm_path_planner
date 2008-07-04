/**
 * 
 */
package sibtra.lms;

/**
 * Clase con una serie de utilidades para el manejo de mensajes del LSM
 * 
 * @author alberto
 *
 */
public class UtilMensajes extends ParametrosLMS {

	/**
	 * Código del rango de distancias.
	 * Según está definido en bloque D del comando 77 (pag. 98)
	 */
	private byte codigoRango=00;


	/**
	 * Se inicializan los paramentros para interpretar los mensajes.
	 *  
	 * @param resAngular
	 * @param rangoAngular
	 * @param enMilimetros
	 * @param codigoRango
	 */
	public UtilMensajes(short resAngular, short rangoAngular,
			boolean enMilimetros, byte codigoRango) {
		super(rangoAngular,resAngular,enMilimetros);
		setCodigoRango(codigoRango);
	}

	/**
	 * Devuelve el código de rango configurado.
	 * @return the codigoRango
	 */
	public byte getCodigoRango() {
		return codigoRango;
	}

	/**
	 * Modifica el código de rango configurado.
	 * Valores soportados: de 0 a 6 
	 * @param codigoRango the codigoRango to set
	 */
	public void setCodigoRango(byte codigoRango) {
		if(codigoRango>=0 && codigoRango<=6)
			this.codigoRango = codigoRango;
	}

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
	 * Maneja un mensaje y devuelve un barrido angular.
	 * Que los campos de bits superiores están a 0. Hay que mejorar si no es el caso.
	 * @param mensaje Mensaje a interpretar.
	 * @return Barrido angular correspondiente. null si hay algún error
	 */
	public BarridoAngular mensajeABarridoAngular(byte[] mensaje) {
		// Nos aseguramos que tiene al menos el tipo
		if(mensaje==null || mensaje.length<1)
			return null;
		//Cosas particulares de cada tipo
		int campoNumDatos=0;
		int aIni=(getRangoAngular()==100?40*4:0)  //0, si la resolución es 100 => empezamos en 40º
			, aFin=(getRangoAngular()==100?140*4:180*4)	//180, si res angular es 100 => terminamos en 140º
			, aInc=getResAngularCuartos() //resolución según está configurada en 1/4 de º
			, indDatos=3	//indice donde comienzan los datos
			;
		short promedios=1	//por defecto promedio de 1 barrido
			;
		switch(mensaje[0]) {
		case (byte) 0xb0:
			if(mensaje.length<3) return null;
			campoNumDatos=men2Word(mensaje,1);
			break;
		case (byte) 0xb7:
			if(mensaje.length<7) return null;
			aIni=(men2Word(mensaje, 1)-1)*4/aInc+(getRangoAngular()==100?40*4:0);
			aFin=(men2Word(mensaje, 3)-1)*4/aInc+(getRangoAngular()==100?40*4:0);
			campoNumDatos=men2Word(mensaje,5);
			indDatos=7;
			break;
		case (byte) 0xb6:
			if(mensaje.length<4) return null;
			promedios=(short) (mensaje[1]&0xff); //para no extender el signo
			campoNumDatos=men2Word(mensaje,2);
			indDatos=4;
			break;
		case (byte) 0xbf:
			if(mensaje.length<8) return null;
			promedios=(short) (mensaje[1]&0xff); //para no extender el signo
			aIni=(men2Word(mensaje, 2)-1)*4/aInc+(getRangoAngular()==100?40*4:0);
			aFin=(men2Word(mensaje, 4)-1)*4/aInc+(getRangoAngular()==100?40*4:0);
			campoNumDatos=men2Word(mensaje,6);
			indDatos=8;
			break;
		default:
			//no es tipo soportado
			return null;
		}
		int numDatos=campoNumDatos&(0x1ff); //numero del datos del barrido: bits del 0 a 9
		int codParcial=campoNumDatos&(0x03<<11)>>11; //que barrido parcial es
		boolean esParcial=(campoNumDatos&(1<<13))==1;
		int mDist=(campoNumDatos&(0x03<<14))>>14; //codigo para las unidades de medida (cm , mm otro)
		
		if(((aFin-aIni)/aInc+1)!=numDatos)
			//numero de datos no es correcto
			return null;
		
		if(indDatos+numDatos*2+1>mensaje.length)
			//mensaje no tiene el tamaño suficiente
			return null;
		
		if(mDist>1)
			return null; //codigo de distancia reservado
		
		if(esParcial) {
			aIni=aIni+codParcial;  //TODO No está claso en el caso de subrango parcial??
			aInc=1;  //el parcial es siempre en 1/4 de grado.
		} 

		BarridoAngular ba=new BarridoAngular(numDatos,aIni,aInc,codigoRango,(mDist==1),promedios);
		//nos limitamos a copiar los datos
		for(int i=0; i<numDatos; i++)
			ba.datos[i]=(short)UtilMensajes.men2Word(mensaje, i*2+indDatos);
		return ba;
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
		
		distancias[0]=(double)UtilMensajes.men2Word(mensaje, 3)*((mDist==0)?10.0:1.0);
		distancias[1]=(double)UtilMensajes.men2Word(mensaje, 5)*((mDist==0)?10.0:1.0);
		distancias[2]=(double)UtilMensajes.men2Word(mensaje, 7)*((mDist==0)?10.0:1.0);
		
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
	 * Calcula el CRC para el buffer
	 * @param buf donde se encuentra el mensaje
	 * @param Len se consideran lon len primeros bytes.
	 * @return CRC calculado
	 */
	public static int CalculaCRC(byte[] buf,int Len) {
	
		//#define CRC16_GEN_POL 0x8005
	
		int uCRC16;
		int i;
	
		uCRC16=buf[0];
		i=0;
		while(i<Len-1) {
			if((uCRC16 & 0x8000)!=0) {
				uCRC16=(uCRC16 & 0x7fff)<<1;
				uCRC16^=0x8005;
			} else 
				uCRC16<<=1;
	
			uCRC16 ^= ((int)buf[i+1]&0xff) | (((int)buf[i]&0xff)<<8) ;
			uCRC16 &=0xffff; //quitamos desbordamiento
			i++;
		}
	
		return uCRC16;
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
}
