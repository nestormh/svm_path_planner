/**
 * 
 */
package sibtra.lms;

import java.io.Serializable;

import java.awt.geom.Point2D;


/**
 * @author alberto
 *
 */
public class ZonaSegmentadaLMS extends ZonaLMS implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 146321635140243289L;

	private short numeroSegmentos;

	public short[] radiosPuntos;
	
	/**
	 * Constructor por defecto para que sea serializable.
	 */
	public ZonaSegmentadaLMS() {
		
	}
	
	/**
	 * 
	 * @param rangoAngular
	 * @param resAngular
	 * @param enMilimetros
	 * @param conjunto1
	 * @param queZona
	 */
	public ZonaSegmentadaLMS(short rangoAngular, short resAngular, boolean enMilimetros, boolean conjunto1, byte queZona) {
		super(rangoAngular,resAngular,enMilimetros,conjunto1,queZona);
		numeroSegmentos=0;
		radiosPuntos=null;
	}

	/**
	 * 
	 * @param rangoAngular
	 * @param resAngular
	 * @param enMilimetros
	 * @param conjunto1
	 * @param queZona
	 * @param numeroSegmentos
	 */
	public ZonaSegmentadaLMS(short rangoAngular, short resAngular, boolean enMilimetros, boolean conjunto1, byte queZona
			,short numeroSegmentos) {
		super(rangoAngular,resAngular,enMilimetros,conjunto1,queZona);
		if(!setNumeroSegmentos(numeroSegmentos)) {
			//si el número de segmentos no es correcto lo inicializamos a 0
			numeroSegmentos=0;
			radiosPuntos=null;
		}
	}

	/**
	 * @return the numeroSegmentos
	 */
	public short getNumeroSegmentos() {
		return numeroSegmentos;
	}

	/**
	 * Estamblecemos el número de segmentos y definimos el array para contenerlos.
	 * El número de segmentos tiene restricciones.
	 * Si el rango angular es 180º sólo se permiten: 9, 10, 15, 18, 30, 45, 90, 180, 360
	 * Si el rango angular es 100º sólo se permiten: 5, 10, 50, 100, 400
	 * Permitimos también los 200 en 100º para el caso taugh-in de resolución 0.5º
	 * @param nS the numeroSegmentos to set
	 * @return si el valor era válido y se pudo actualizar
	 */
	public boolean setNumeroSegmentos(short nS) {
		if( 
				( (getRangoAngular()==180) &&
						(nS==9 || nS==10 || nS==15 || nS==18 || nS==30 || nS==45 || nS==180 || nS==360 ) )
						||
						( (getRangoAngular()==100) &&
								(nS==5 || nS==10 || nS==50 || nS==100 || nS==200 || nS==400 ) )
		) {
				numeroSegmentos=nS;
				radiosPuntos=new short[numeroSegmentos+1]; 
				return true;
			}
		return false;				
	}

	public static ZonaSegmentadaLMS MensajeC5AZona(byte[] men) {
		//Primeras comprobaciones de tipo y tamaño
		if(men==null || men.length<=13 || men[0]!=((byte)0xc5) )
			return null;
		boolean conjunto1=(men[1]==1);
		boolean enMilimetros=(((int)men[2]&0xc0)>>6)==1;
		byte zona;
		boolean esSegmentada;
		//Tenemos que decidir zona y si es segmentada o taught-in
		switch ((men[2]&0x0f)) {
		case 2:
			zona=ZONA_A; esSegmentada=true; break;
		case 3:
			zona=ZONA_A; esSegmentada=false; break;
		case 8:
			zona=ZONA_B; esSegmentada=true; break;
		case 9:
			zona=ZONA_B; esSegmentada=false; break;
		case 14:
			zona=ZONA_C; esSegmentada=true; break;
		case 15:
			zona=ZONA_C; esSegmentada=false; break;

		default:
			return null;
		}
		ZonaSegmentadaLMS nz=new ZonaSegmentadaLMS((short)UtilMensajes.men2Word(men, 3)
				,(short)UtilMensajes.men2Word(men, 5)
				,enMilimetros
				,conjunto1
				,zona
				);
		short numSegmentos; //numero de puntos de la zona
		int indDatos;  //indice donde se encuentra el primer dato
		if(esSegmentada) {
			numSegmentos=(short)((int)(men[13])&0xff);
			if(men[13]==(byte)0xFE) 
				numSegmentos=360;
			if(men[13]==(byte)0xFF)
				numSegmentos=400;
			if(!nz.setNumeroSegmentos(numSegmentos))
				//el número de segmentos no es válido
				return null;
			indDatos=14; //donde comienzan distancias
		} else {
			//es zona taught-in
			numSegmentos=(short)(nz.getRangoAngular()*4/nz.getResAngularCuartos());
			if(!nz.setNumeroSegmentos(numSegmentos))
				//el número de segmentos no es válido
				return null;
			indDatos=13;
		}
		//Vemos si los datos son suficientes
		if(men.length<(numSegmentos+1)*2+indDatos) 
			return null;
		nz.radiosPuntos=new short[numSegmentos+1];
		for(int nd=0; nd<(numSegmentos+1); nd++, indDatos+=2)
			nz.radiosPuntos[nd]=(short)UtilMensajes.men2Word(men, indDatos);
		return nz;
	}
	
	public pointIterator creaPointIterator() {
		return new pointIterator();
	}
	
	public class pointIterator {
		
		/**
		 * Indice del punto actual
		 */
		int indPtoAct;
		
		/**
		 * Angulo del punto actual en 1/4 de grado
		 */
		long angAct;
		
		/**
		 * Incremento angular a aplicar en 1/4 de grado
		 */
		long incAng;
		
		/**
		 * Constructor del iterator
		 */
		protected pointIterator() {
			indPtoAct=0;
			angAct=((getRangoAngular()==180)?0:40)*4; //donde comienzan los ángulo
			//el incremento angular será del rango angular entre el número de segmentos
			incAng=getRangoAngular()*4/numeroSegmentos;
		}
		
		/** @return Indica si quedan más puntos */
		public boolean hasNext() {
			return indPtoAct<radiosPuntos.length;
		}
		
		/**
		 * @return Siguiente punto de la zona
		 */
		public Point2D.Double next() {
			if(indPtoAct>=radiosPuntos.length)
				return null;
			double ang=Math.toRadians((double)angAct/4); //lo pasamos a radianes
			 //lo pasamos a metros
			double dis=(double)((int)radiosPuntos[indPtoAct]&0xffff)/(isEnMilimetros()?1000.0:100.0);
			//incrementamos valores antes de volver
			indPtoAct++;
			angAct+=incAng;
			return new Point2D.Double((double)(dis*Math.cos(ang))
					,(double)(dis*Math.sin(ang)));
		}
	}
}
