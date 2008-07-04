/*
 * Creado el 18/02/2008
 *
 *  Creado por Alberto Hamilcon con Eclipse
 */
package sibtra.lms;

import java.awt.geom.Point2D;

/**
 * @author alberto
 *
 * Objeto que contendrá los datos de un barrido angular.
 * Almacenamos vector de datos como los manda el LSM. 
 * Necesitamos saber el codigo-rango utilizado para separar datos de flags.
 * Si promedios > 1 se supone que no hay flags (aunque es necesario el codigo-rango para saber tamaño de los datos)  
 * Tambien necitamos el angulo inicial, el incremento angular y si está en milímetros.
 *<p> 
 * Se define iterador para obtener valores procesados a partir de los datos: puntos, angulos, distancias, 
 * zonas infringidas, etc.
 */
public class BarridoAngular implements java.io.Serializable {
	/**
	 * No inicializa los arrays de angulo y distancia
	 */
	public BarridoAngular() {
		datos=null;
		promedios=1;
		enMilimetros=false;
	}

	/**
	 * Inicializa los arrays de angulo y distancia la tamaño indicado
	 * @param numDatos
	 */
	public BarridoAngular(int numDatos) {
		datos=new short[numDatos];
		promedios=1;
		enMilimetros=false;
	}

	/**
	 * Contrunstor completo
	 * @param numDatos
	 * @param anguloInicial en 1/4 de grado
	 * @param incAngular en 1/4 de grado
	 * @param codigoRango de 0 a 6
	 * @param enMilimetros
	 * @param promedios Si tenemos más de un promedio sabemos que no hay flags
	 */
	public BarridoAngular(int numDatos,
			int anguloInicial, int incAngular, byte codigoRango, boolean enMilimetros, short promedios) {
		datos=new short[numDatos];
		this.promedios = promedios;
		this.enMilimetros = enMilimetros;
		this.incAngular = incAngular;
		this.anguloInicial = anguloInicial;
		this.codigoRango = codigoRango;
	}



	/**
	 * Datos (WORD) como se reciben del LSM.
	 * Contendrán distancia y otros flags dependiendo del código de rango en cada momento.
	 * Para acceder a los datos tratados se debe utilizar el iterador.
	 */
	public short	datos[];
	
	/**
	 * Numero de promedios que se har realizado para obterner la distancia
	 */
	public short	promedios;
	
	/**
	 * Si es verdadero la distancia es en milímetros, si es falso en centímetros
	 */
	public boolean	enMilimetros;

	/**
	 * Incremento angular en 1/4 de grado.
	 */
	public int incAngular;
	
	/**
	 * Angulo inicial en 1/4 de grado
	 */
	public int anguloInicial;
	
	/**
	 * Codigo de rángo. Según se definen en Bloque D de mensaje 77
	 */
	byte codigoRango;

	
	/**
	 * @return Número de datos del barrido.
	 */
	public int numDatos() {
		return datos.length;
	}
	/**
	 * @return máscara con los bits que contienen los datos
	 */
	public short mascaraDatos() {
		short mascaraDatos=0x7fff;  //los bits del 0 a 14
		if(codigoRango<=2)
			mascaraDatos=0x1fff; //los bits del 0 al 12
		else if(codigoRango<=5)
			mascaraDatos=0x3fff; //los bits del 0 al 13
		return mascaraDatos;
	}

	/**
	 * @return el bit donde está la Zona A en los datos. 0 si no está.
	 */
	public short bitsDeA() {
		if(promedios>1) return 0;
		switch(codigoRango) {
		case 0:
		case 2:
			return (1<<13);
		case 4:
			return (1<<14);
		case 6:
			return (short)(1<<15);
		default:
			return 0;
		}
	}

	/**
	 * @return el bit donde está la Zona B en los datos. 0 si no está.
	 */
	public short bitsDeB() {
		if(promedios>1) return 0;
		switch(codigoRango) {
		case 0:
		case 2:
			return (1<<14);
		case 4:
			return (short)(1<<15);
		default:
			return 0;
		}
	}

	/**
	 * @return el bit donde está la Zona C en los datos. 0 si no está.
	 */
	public short bitsDeC() {
		if(promedios>1) return 0;
		switch(codigoRango) {
		case 2:
			return (short)(1<<15);
		default:
			return 0;
		}
	}

	/**
	 * @return true si el barrido tienen información de la zona
	 */
	public boolean hayInformacionZonaA() {
		return bitsDeA()!=0;
	}

	/**
	 * @return true si el barrido tienen información de la zona
	 */
	public boolean hayInformacionZonaB() {
		return bitsDeB()!=0;
	}

	/**
	 * @return true si el barrido tienen información de la zona
	 */
	public boolean hayInformacionZonaC() {
		return bitsDeC()!=0;
	}

	/**
	 * @return true si el barrdo infringe la zona. false si no infringe o NO HAY INFORMACIÓN
	 */
	public boolean infringeA() {
		short bdz=bitsDeA();
		if (bdz==0)
			return false;
		boolean infringido=false;
		for(int i=0; !infringido && i<datos.length; i++)
			infringido=(datos[i]&bdz)!=0;
		return infringido;
	}

	/**
	 * @return true si el barrdo infringe la zona. false si no infringe o NO HAY INFORMACIÓN
	 */
	public boolean infringeB() {
		short bdz=bitsDeB();
		if (bdz==0)
			return false;
		boolean infringido=false;
		for(int i=0; !infringido && i<datos.length; i++)
			infringido=(datos[i]&bdz)!=0;
		return infringido;
	}

	/**
	 * @return true si el barrdo infringe la zona. false si no infringe o NO HAY INFORMACIÓN
	 */
	public boolean infringeC() {
		short bdz=bitsDeC();
		if (bdz==0)
			return false;
		boolean infringido=false;
		for(int i=0; !infringido && i<datos.length; i++)
			infringido=(datos[i]&bdz)!=0;
		return infringido;
	}
	
	public barridoAngularIterator creaIterator() {
		return new barridoAngularIterator();
	}
	/** @return Angulo del dato i-ésimo EN RADIANES */	
	public double getAngulo(int i) {
		if (i<0 || i>=datos.length) return Double.NaN;
		return Math.toRadians((anguloInicial+i*incAngular)*0.25);		
	}
	
	/** @return Distancia correspondiente al dato i-ésimo */
	public double getDistancia(int i) {
		if (i<0 || i>=datos.length) return Double.NaN;
		return (double)(datos[i]&mascaraDatos())/(enMilimetros?1000:100);
	}

	/** @return {@link Point2D} correspondiente al dato i-ésimo */ 
	public Point2D.Double getPunto(int i){
		if (i<0 || i>=datos.length) return null;
		double ang=getAngulo(i);
		double dis=getDistancia(i);
		return new Point2D.Double(dis*Math.cos(ang),dis*Math.sin(ang));
	}
	
	public double getDistanciaMaxima() {
		if(codigoRango==2 && enMilimetros)
			return 8.0;
		if(codigoRango==4 && enMilimetros)
			return 16.0;
		if(codigoRango==6 && enMilimetros)
			return 32.0;
//		if(codigoRango==2 && !enMilimetros)
//			return 80.0;
		return 80.0;
	}
	
	/**
	 * Iterador que nos permitirá obtener los datos que hay en el barrido angular.
	 * next() devuelve true mientras queden elementos. La forma de usar sería
	 * <code>
	 * barridoAngularIterator it=ba.creaIterator();
	 * while(it.next()) {
	 *   it.punto();
	 *   it.angulo();
	 *   it.infringeA();
	 *   ...
	 * } 
	 *</code>
	 * @author alberto
	 *
	 */
	public class barridoAngularIterator {
		
		/**
		 * Indice del punto actual
		 */
		private int indPtoAct;
		
		/**
		 * Angulo del punto actual en 1/4 de grado
		 */
		private long angAct;
		
		private short bitsDeA;
		private short bitsDeB;
		private short bitsDeC;

		private short mascaraDatos;
		
		/**
		 * Constructor del iterator
		 */
		protected barridoAngularIterator() {
			//empezamos en -1
			indPtoAct=-1;
			angAct=anguloInicial-incAngular; //donde comienzan los ángulo
			//Cacheamos máscara de bits
			bitsDeA=bitsDeA();
			bitsDeB=bitsDeB();
			bitsDeC=bitsDeC();
			mascaraDatos=mascaraDatos();
		}
		
		/** @return Indica si quedan más puntos	 */
		public boolean hasNext() {
			return indPtoAct<datos.length;
		}
		
		/**
		 * Pasa al siguiente punto
		 * @return true si se pasó, false si no quedan más.
		 */
		public boolean next() {
			if (indPtoAct>=(datos.length-1))
				return false;
			indPtoAct++;
			angAct+=incAngular;
			return true;
		}
		
		/**
		 * @return Siguiente punto de la zona
		 */
		public Point2D.Double punto() {
			if (indPtoAct<0 || indPtoAct>=datos.length) return null;
			double ang=Math.toRadians((double)angAct/4); //lo pasamos a radianes
			 //lo pasamos a metros
			double dis=(double)(datos[indPtoAct]&mascaraDatos)/(enMilimetros?1000.0:100.0);
			return new Point2D.Double(dis*Math.cos(ang),dis*Math.sin(ang));
		}

		/**
		 * @return ángulo del punto actual en grados
		 */
		public double angulo() {
			return (double)angAct/4.0;
		}

		/**
		 * 
		 * @return distancia del punto actual en metros
		 */
		public double distancia() {
			if (indPtoAct<0 || indPtoAct>=datos.length) return Double.NaN;
			return (double)(datos[indPtoAct]&mascaraDatos())/(enMilimetros?1000:100);
		}
		
		/**
		 * @return true si el punto actual infringe la zona A. false si no infringe o NO HAY INFORMACIÓN
		 */
		public boolean infringeA() {
			if (indPtoAct<0 || indPtoAct>=datos.length) return false;
			if(bitsDeA==0)
				return false;
			return (datos[indPtoAct]&bitsDeA)==1;
		}
		
		/**
		 * @return true si el punto actual infringe la zona B. false si no infringe o NO HAY INFORMACIÓN
		 */
		public boolean infringeB() {
			if (indPtoAct<0 || indPtoAct>=datos.length) return false;
			if(bitsDeB==0)
				return false;
			return (datos[indPtoAct]&bitsDeB)==1;
		}
		
		/**
		 * @return true si el punto actual infringe la zona C. false si no infringe o NO HAY INFORMACIÓN
		 */
		public boolean infringeC() {
			if (indPtoAct<0 || indPtoAct>=datos.length) return false;
			if(bitsDeC==0)
				return false;
			return (datos[indPtoAct]&bitsDeC)==1;
		}
	}
}
