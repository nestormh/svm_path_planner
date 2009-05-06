/**
 * 
 */
package sibtra.lms;

import java.util.Arrays;

/**
 * @author alberto
 *
 */
public class ManejaLMS {

	/**
	 * Password del LMS
	 */
	protected static String LMSPassword="SICK_LMS";
	//TODO que el password se envie o configure de alguna manera dinámica

	/**
	 * Objeto para la comunicación serial con el LMS
	 */
	protected ManejaTelegramas	manTel;

	/**
	 * Se encarga de manejar y convertir los mensajes.
	 * Contendrá de manera actualizada los parámetros que (se supone) tiene configurado el LMS.
	 */
	protected UtilMensajes	manMen;

	private boolean lmsVivo=false;
	
	private boolean configuradoCodigo=false;

	private boolean configuradoVariante=false;
	
	/**
	 * variable para saber en que estado estamos. Si estamos a la espera de alguna respuesta. 
	 */
	private int pidiendo;
	private static final int PIDIENDO_NADA=0;
//	private static final int PIDIENDO_ZONA=1;
	private static final int PIDIENDO_BARRIDO=2;
	private static final int PIDIENDO_DISTANCIAS=3;
	private static final int PIDIENDO_CONTINUO=4;
	private static final int PIDIENDO_PARAR=5;

	/** milisegundos a esperar antes de purgar buffer y mandar mensajes.
	 * Sólo se aplica a mensajes de configuración.
	 */
	private static final long milisPreviosEnvio = 500;

	/** Thread encargado de recibir las peticiones continuas */
	private ThreadContinuo thContinuo;
	
	/** Mutex para el acceso a los barridos continuos */
	private Object mutexBarridos=new Object();

	/** Contendrá el último barrido continuo recibido */
	private BarridoAngular ultimoBarrido=null;
	
	/** numero de reintentos en el envío de un mensaje */
	private int numReintentos=4; 
	
	/**
	 * Constructor, por defecto usa ManejaTelegramasJNI
	 * @param puertoSerie
	 */	
	public ManejaLMS(String puertoSerie){
		this(puertoSerie, new ManejaTelegramasJNI());
	}
	
	/**
	 * Constructor
	 * @param puertoSerie
	 * @param mantel objeto de manejo de telegramas a usar
	 */
	public ManejaLMS(String puertoSerie, ManejaTelegramas mantel) {
		manTel=mantel;
		manTel.ConectaPuerto(puertoSerie);
		manTel.setDefaultTimeOut(500);
		
		//valores por defecto supuestos
		manMen=new UtilMensajes((short)50,(short)180,true,(byte)0);

		pidiendo=PIDIENDO_NADA;
		
		lmsVivo=false;
		configuradoVariante=false;
		configuradoCodigo=false;

		thContinuo=new ThreadContinuo();
		thContinuo.start(); //comienza suspendido
		
		try { configuraInicial(); } 
		catch (LMSException e) {
			System.err.println("ERROR: "+e.getMessage());
		};
		
		
	}
	
	/**
	 * Se encarga de verificar la comunicació con el RF y ponerla 500Kba 
	 * @return si fue posible fijar y confirmar el paso
	 */
	public void configura500K() throws LMSException {
		configura500K(4);
	}

	/**
	 * Se encarga de verificar la comunicació con el RF y ponerla 500Kba
	 * @param maxIntentos numero de intentos en para cada mensaje enviado 
	 * @return si fue posible fijar y confirmar el paso
	 */
	public void configura500K(int maxIntenos) throws LMSException {
		lmsVivo=false;
		if (!manTel.isInicializado())
			throw new LMSException("Puerto aún no inicializado");
		byte[] MenPasoA25={0x20, 0x25}; //aprovechamos para ir parando envío continuo
		byte[] Men500k={0x20, 0x48};
		System.out.println("Tratamos de configurar a 500Kb");
		for(int ia=1; ia<=maxIntenos; ia++) {
			manTel.setBaudrate(38400);
			boolean rOK=false;
			for(int i=1; !rOK && i<=maxIntenos; i++) {
				try { Thread.sleep(milisPreviosEnvio); } catch (InterruptedException e) {}
				manTel.purgaBufferEntrada();
				manTel.EnviaMensaje(MenPasoA25);
				rOK=(manTel.LeeMensaje()!=null);
			}
			if(rOK) {
				System.out.println("RF responde bien a 38400");
				lmsVivo=true;
				return;
			}
			System.err.println("No se recibió bien el mensaje a 38400. pasamos a 9600");
			manTel.setBaudrate(9600);
			rOK=false;
			for(int i=1; !rOK && i<=maxIntenos; i++) {
				try { Thread.sleep(milisPreviosEnvio); } catch (InterruptedException e) {}
				manTel.purgaBufferEntrada();
				manTel.EnviaMensaje(MenPasoA25,1000); //mas timeout ya que es 9600
				rOK=(manTel.LeeMensaje(2000)!=null); //mas timeout ya que es 9600
			}
			if(!rOK) {
				System.err.println("RF no responde tampoco a 9600, hacemos otro intento desde el principio");
				continue;
			}
			System.err.println("Esta a 9600, trataremos de pasarlo a 500K");
			rOK=false;
			for(int i=1; !rOK && i<=maxIntenos; i++) {
				try { Thread.sleep(milisPreviosEnvio); } catch (InterruptedException e) {}
				manTel.purgaBufferEntrada();
				manTel.EnviaMensaje(Men500k,1000); //mas timeout ya que es 9600
				rOK=(manTel.LeeMensaje(2000)!=null); //mas timeout ya que es 9600
			}
			try { Thread.sleep(2000); } catch (InterruptedException e) {}
			System.err.println("haya ido bien o mal volvemos al principio a intentarlo a 38400");
		}
		throw new LMSException("No fue posible configurar a 500Kba");
	}
	
	/** @return si es se ha conseguido conectar con el LMS */
	public boolean isVivo() {
		return lmsVivo;
	}

	/** Tranta de conectar con LMS a 500Kba y determinar su configuración */ 
	void configuraInicial() throws LMSException {
		
        // Lo pasamos a 500K si no lo está aún
		if(!lmsVivo)
			configura500K();
		if(!configuradoCodigo) {
				byte[] respConfig;
				respConfig=ObtieneParte1Configuracion();
				manMen.setCodigoRango(respConfig[6]);
				manMen.setEnMilimetros(respConfig[7]==1);
				configuradoCodigo=true;
			}
		if(!configuradoVariante) {
			setVariante(manMen.getRangoAngular(),manMen.getResAngular());
			configuradoVariante=true;
		}
	}
	
	/**
	 * Manda los mensajes para cambiar el LMS a modo 25.
	 * En este modo el LMS no envía nada sino espera peticiones.
	 * @throws LMSException  si hay problemas en la comunicación.
	 */
	public void CambiaAModo25() throws LMSException {
		if(!lmsVivo) throw new LMSException("LMS no esta vivo");
		if(pidiendo!=PIDIENDO_NADA)
			throw new LMSException("Estamos en medio de una peticion");
		System.out.println("Tratamos de pasar a modo 25");
		//pasamos al modo 25
		byte[] menModo25={0x20, 0x25};
		boolean menOK=false;
		for(int ni=1; !menOK && ni<=numReintentos; ni++) {
			try { Thread.sleep(milisPreviosEnvio); } catch (InterruptedException e) {}
			manTel.purgaBufferEntrada();
			manTel.EnviaMensaje(menModo25); 
			byte[] respModo25;
			if(( respModo25=manTel.LeeMensaje())==null)
				continue; //no hay respuesta volvemos a intentarlo

			byte[] respModo25OK={(byte)0xa0, 00, 0x10};
			menOK=Arrays.equals(respModo25,respModo25OK);
		}
		if(!menOK)
			throw new LMSException("No fue posible el cambio al modo 25 tras "+numReintentos
					+" reintentos");
		System.out.println("Conseguido el paso a modo 25");
	}
	/**
	 * Manda los mensajes para cambiar a modo 00 (instalación)
	 * @throws LMSException  si hay problemas en la comunicación.
	 */
	protected void CambiaAModoInstalacion() throws LMSException {
		if(!lmsVivo) throw new LMSException("LMS no esta vivo");
		//pasamos al modo 00
		System.out.println("Tratamos de pasar a modo instalación");

		byte[] menModo00=new byte[LMSPassword.length()+2];
		menModo00[0]=0x20; menModo00[1]=0;
		for(int i=0; i<LMSPassword.length();i++)
			//la conversión debe funcionar ya que caracteres ASCII sólo usan 7 bits.
			menModo00[i+2]=(byte)LMSPassword.charAt(i);

		//Hacemos bucle de reintentos
		boolean menOK=false;
		for(int ni=1; !menOK && ni<=numReintentos; ni++) {
			try { Thread.sleep(milisPreviosEnvio); } catch (InterruptedException e) {}
			manTel.purgaBufferEntrada();
			manTel.EnviaMensaje(menModo00);
			byte[] respModo00;
			if(( respModo00=manTel.LeeMensaje())==null)
				continue; //no hay respuesta volvemos a intentarlo
			byte[] respModo00OK={(byte)0xa0, 00, 0x10};
			menOK=Arrays.equals(respModo00,respModo00OK);
		}
		if(!menOK)
			throw new LMSException("No fue posible el cambio al modo instalación ");
		System.out.println("Conseguido paso a modo instalación");
	}
	
	/**
	 * Obtiene la parte 1 de la configuración a través de mensaje 74
	 * @return Mensaje entero de respuesta f4, incuyendo byte de status.
	 * @throws LMSException  si hay problemas en la comunicación.
	 */
	protected byte[] ObtieneParte1Configuracion() throws LMSException {
		if(!lmsVivo) throw new LMSException("LMS no esta vivo");
		if(pidiendo!=PIDIENDO_NADA)
			throw new LMSException("Estamos en medio de una peticion");
		System.out.println("Tratamos de conseguir parte 1 de la configuración");

		byte[] menMiraConfiguracion={0x74};
		byte[] respConfig=null;
		boolean menOK=false;
		for(int ni=1; !menOK && ni<=numReintentos; ni++) {
			try { Thread.sleep(milisPreviosEnvio); } catch (InterruptedException e) {}
			manTel.purgaBufferEntrada();
			manTel.EnviaMensaje(menMiraConfiguracion);
			if( (respConfig=manTel.LeeMensaje())==null)
				continue; //no hay respuesta volvemos a intentarlo
			menOK= (respConfig.length==36) && (respConfig[0]==(byte)0xf4);
		}
		if(!menOK)
			throw new LMSException("Respuesta incorrecta a la consulta de configuración ");
		System.out.println("Conseguida parte 1 de la configuración");
		return respConfig;

	}
	
	/**
	 * Envía mensaje de cambio de configuración de parte 1
	 * @param menConfigura mensaje adecuado (77) para el cambi 
	 * @throws LMSException  si hay problemas en la comunicación.
	 */
	protected void CambiaParte1Configuracion(byte[] menConfigura) throws LMSException {
		if(!lmsVivo) throw new LMSException("LMS no esta vivo");
		if(pidiendo!=PIDIENDO_NADA)
			throw new LMSException("Estamos en medio de una peticion");

		System.out.println("Tratamos de cambiar la parte 1 de la configuración");
		boolean menOK=false;
		for(int ni=1; !menOK && ni<=numReintentos; ni++) {
			try { Thread.sleep(milisPreviosEnvio); } catch (InterruptedException e) {}
			manTel.purgaBufferEntrada();
			manTel.EnviaMensaje(menConfigura); 
			byte [] respNuevaConfig;
			if( (respNuevaConfig=manTel.LeeMensaje())==null)
				continue; //no hay respuesta volvemos a intentarlo
			menOK=UtilMensajes.esConfirmacion(respNuevaConfig, menConfigura);
		}
		if(!menOK)
			throw new LMSException("No se aceptó la nueva configuración");
		System.out.println("Conseguido el cambio de la parte 1 de la configuración");

	}

	
	
	/**
	 * Envia mensaje de cambio de variante según lo pasado.
	 * @param rangoAngular rango angular 180=180º ó 100=100º
	 * @param resAngular resolución angular en decimas de grado 100=1º, 50=0.5º, 25=0.25º
	 * @throws LMSException  si hay problemas en la comunicación.
	 */
	public void setVariante(short rangoAngular, short resAngular) throws LMSException {
		if(configuradoVariante && manMen.getRangoAngular()==rangoAngular && manMen.getResAngular()==resAngular)
			return; //estamos en la variante pedida

		if(rangoAngular!=180 && rangoAngular!=100)
			throw new IllegalArgumentException("Parámetros de rango angular ("+rangoAngular+") incorrecto");

		if(resAngular!=100 && resAngular!=50 && resAngular!=25)
			throw new IllegalArgumentException("Parámetros de resolución angular ("+resAngular+") incorrecto");
		
		if(pidiendo!=PIDIENDO_NADA)
			throw new LMSException("Estamos en medio de una peticion");
		System.out.println("Tratamos de fijar la variante");

		if(!lmsVivo) throw new LMSException("LMS no esta vivo");
		final byte[] menVariante={(byte)0x3b, (byte)0xb4, 0, 0x32, 0};  //Mensaje ejemplo
		UtilMensajes.word2Men(rangoAngular,menVariante,1);  //valores actuales
		UtilMensajes.word2Men(resAngular, menVariante, 3);

		boolean menOK=false;
		for(int ni=1; !menOK && ni<=numReintentos; ni++) {
			try { Thread.sleep(milisPreviosEnvio); } catch (InterruptedException e) {}
			manTel.purgaBufferEntrada();
			manTel.EnviaMensaje(menVariante); 
			byte[] respVariante;
			if((respVariante=manTel.LeeMensaje())==null)
				continue; //volvemos a intentarlo
			menOK=UtilMensajes.esConfirmacion(respVariante,menVariante);
		}
		if(!menOK) 
			throw new LMSException("No fue posible el cambio de variante ");
		//Ponemos rango 180º y 1/2 grado de resolucion
		manMen.setRangoAngular(rangoAngular);
		manMen.setResAngular(resAngular); 
		configuradoVariante=true;
		System.out.println("Se ha conseguido fijar la variante");

	}
	
	/**
	 * Trata de hacer cambio en LMS del rango angular
	 * @param rangoAngular  rango angular 180=180º ó 100=100º
	 * @throws LMSException si hay problemas en la comunicación.
	 */
	public void setRangoAngular(short rangoAngular) throws LMSException {
		setVariante(rangoAngular, manMen.getResAngular());
	}
	
	/**
	 * Devuelve rango angular
	 * @return en grados: 180 ó 100
	 * @throws LMSException si hay problemas en la comunicación.
	 */
	public short getRangoAngular() throws LMSException {
		configuraInicial();
		if(!configuradoVariante)
			throw new LMSException("Variante no configurada");
		return manMen.getRangoAngular();
	}
	
	/**
	 * Trata de hacer cambio en LMS de la resolución angular
	 * @param resAngular resolución angular en decimas de grado 100=1º, 50=0.5º, 25=0.25º
	 * @throws LMSException si hay problemas en la comunicación.
	 */
	public void	setResolucionAngular(short resAngular) throws LMSException {
		setVariante(manMen.getRangoAngular(), resAngular);
	}
	
	/**
	 * Devuelve resolucion angular
	 * @return resolucion angular en centesimas de grado: 100, 50 ó 25
	 * @throws LMSException si hay problemas en la comunicación.
	 */
	public short getResolucionAngular() throws LMSException {
		configuraInicial();
		if(!configuradoVariante)
			throw new LMSException("Variante no configurada");
		return manMen.getResAngular();
	}
	
	/**
	 * Trata de hacer cambio en LMS de la resolución (mm/cm) y del código de rango
	 * @param enMilimetros true para resolución de milímetros, false para resolución de cm
	 * @param codigoRango valor entre 0 y 6 (según página 98 del manual LMS)
	 * @throws LMSException si hay problemas en la comunicación.
	 */
	public void setCodigo(boolean enMilimetros, byte codigoRango) throws LMSException{
		configuraInicial();
		if(configuradoCodigo && manMen.isEnMilimetros()==enMilimetros && manMen.getCodigoRango()==codigoRango)
			return; //configuración igual a la pedida
		
		if(codigoRango<0 || codigoRango>6)
			throw new IllegalArgumentException("Codigo de rango ("+codigoRango+") fuera del rango permitido de 0 a 6");
		
		//Leemos la configuración acutal porque sólo queremos cambiar dos valores
		byte[] respConfig=ObtieneParte1Configuracion();
		//Lo copiamos en nuevo array ya que hay que quitar el byte de status
		byte[] menConfigura=new byte[respConfig.length-1];
		System.arraycopy(respConfig, 0, menConfigura, 0, menConfigura.length);
		menConfigura[6]=codigoRango;
		menConfigura[7]=(byte)(enMilimetros?1:0);
		menConfigura[0]=(byte)0x77;
		//Tratamos de pasar al modo instalación para poder aplicar la configuración
		CambiaAModoInstalacion();
		try {
			CambiaParte1Configuracion(menConfigura); 
		} catch (LMSException e){
			//antes de lanzar la excepción pasamos a modo 25
			CambiaAModo25();
			throw e;
		}
 		//Todo bien pero debemos volvemos al modo 25
		CambiaAModo25();
		manMen.setCodigoRango(codigoRango);
		manMen.setEnMilimetros(enMilimetros);
		configuradoCodigo=true;
	}
	
	/**
	 * Trata de hacer cambio en LMS de la resolución (mm/cm)
	 * @param enMilimetros true para resolución de milímetros, false para resolución de cm
	 * @throws LMSException si hay problemas en la comunicación.
	 */
	public void setEnMilimetros(boolean enMilimetros) throws LMSException {
		setCodigo(enMilimetros, manMen.getCodigoRango());
	}
	
	/**
	 * Devuelve la resolución en distancia
	 * @return true si es de milímetros false si en cm
	 * @throws LMSException si hay problemas en la comunicación.
	 */
	public boolean isEnMilimetros() throws LMSException {
		configuraInicial();
		if(!configuradoCodigo)
			throw new LMSException("Codigo no configurado");
		return manMen.isEnMilimetros();

	}
	
	/**
	 * Trata de hacer cambio en LMS del código de rango
	 * @param codigoRango valor entre 0 y 6 (según página 98 del manual LMS)
	 * @throws LMSException si hay problemas en la comunicación.
	 */
	public void setCodigoRango(byte codigoRango) throws LMSException {
		setCodigo(manMen.isEnMilimetros(), codigoRango);
	}
	
	/**
	 * @return Devuelve código de rango
	 * @throws LMSException si hay problemas en la comunicación.
	 */
	public byte getCodigoRango() throws LMSException {
		configuraInicial();
		if(!configuradoCodigo)
			throw new LMSException("Codigo no configurado");
		return manMen.getCodigoRango();
	}
	
	/**
	 * Fija alcance
	 * @param metros alcance solicitado, debe ser: 8, 16, 32, 80.
	 * @throws LMSException en caso de cualquier problema
	 */
	public void setDistanciaMaxima(int metros) throws LMSException {
		if (metros==8)
			setCodigo(true, (byte)2);
		else if (metros==16)
			setCodigo(true, (byte)4);
		else if (metros==32)
			setCodigo(true, (byte)6);
		else if (metros==80)
			setCodigo(false, (byte)2);
		else
			throw new IllegalArgumentException("Distancia solicitada ("+metros+") no es 8, 16, 32, 80 ");		
	}
	
	/**
	 * @return el alcance configurado
	 * @throws LMSException  en caso de cualquier problema
	 */
	public int getDistanciaMaxima() throws LMSException {
		if(getCodigoRango()==2 && isEnMilimetros())
			return 8;
		if(getCodigoRango()==4 && isEnMilimetros())
			return 16;
		if(getCodigoRango()==6 && isEnMilimetros())
			return 32;
//		if(getCodigoRango()==2 && !isEnMilimetros())
//			return 80.0;
		return 80;
	}

	/** Solicita zona al LMS
	 * @param queZona 0 para A, 1 para B, 2 para C
	 * @param elConjunto1 true si es conjunto 1, false para conjunto 2
	 * @return la zona recibida
	 * @throws LMSException  en caso de cualquier problema
	 */
	public ZonaLMS recibeZona(byte queZona, boolean elConjunto1) throws LMSException {
		if(!lmsVivo) throw new LMSException("LMS no esta vivo");
		if(pidiendo!=PIDIENDO_NADA)
			throw new LMSException("Estamos en medio de una peticion");
		System.out.println("Pidiendo zona "+queZona+" del conjunto 1:"+elConjunto1);

		//A la vista de los parámetros elegimos el mensaje a mandar
		byte[] men45= {0x45, (byte)(elConjunto1?1:2), queZona};
		ZonaLMS zn=null;
		boolean menOK=false;
		for(int ni=1; !menOK && ni<=numReintentos; ni++) {
			try { Thread.sleep(milisPreviosEnvio); } catch (InterruptedException e) {}
			manTel.purgaBufferEntrada();
			manTel.EnviaMensaje(men45); 
			//Recibimos respuesta
			byte[] respMen=manTel.LeeMensaje(5000); //alargamos TimeOut
			if(respMen==null) 
				continue; //volvemos a reintentarlo
			zn=ZonaLMS.MensajeC5AZona(respMen);
			menOK=(zn!=null);
		}
		if(!menOK)
			throw new LMSException("No se pudo obtener la zona "+queZona
					+" del conjunto uno "+elConjunto1);
		System.out.println("Se consiguió zona "+queZona+" del conjunto 1:"+elConjunto1);
		return zn;
	}
	
	/**
	 * Manda telegrama 30 solicitando las distancias verticales.
	 * @throws LMSException  si hay problemas en la comunicación.
	 */
	public void solicitaDistancia() throws LMSException {
		if(!lmsVivo) throw new LMSException("LMS no esta vivo");
		if(pidiendo!=PIDIENDO_NADA)
			throw new LMSException("Estamos en medio de una peticion");
		final byte[] men30={0x30,0x02}; //Datos de 1 barrido

		if(!manTel.EnviaMensaje(men30))
			throw new LMSException("Error al enviar el mensaje "+UtilMensajes.hexaString(men30));
		pidiendo=PIDIENDO_DISTANCIAS;
	}

	/**
	 * Recive del LMS el telegrama de las distancias vertical de las zonas
	 * @return distancia de cada zona en milímetros
	 * @throws LMSException si hay problemas en la comunicación.
	 */
	public double[] recibeDistancia() throws LMSException {
		if(pidiendo!=PIDIENDO_DISTANCIAS)
			throw new LMSException("NO acabamos de pedir distancias");
		//Recibimos respuesta
		byte[] respMen=manTel.LeeMensaje();
		if(respMen==null)
			throw new LMSException("no se pudo leer mensaje de respuesta.");
		
		double[] distancias=UtilMensajes.mensajeADistancia(respMen);
		if(distancias==null)
			throw new LMSException("no se pudo interpretar correctamente mensaje.");
		return distancias;
	}
	
	/** Cierra la conexión serial. */
	public void cierraPuerto() {
		manTel.cierraPuerto();
	}

	/**
	 * Solicita un barrido al LMS con los parámetros indicados
	 * @param anguloInicial en grados
	 * @param anguloFinal en grados
	 * @param numPromedios
	 * @throws LMSException si se hay algún problema
	 */
	public void pideBarrido(short anguloInicial, short anguloFinal,
			short numPromedios) throws LMSException {
		
		configuraInicial();
		if(pidiendo!=PIDIENDO_NADA)
			throw new LMSException("Estamos en medio de una peticion: "+pidiendo);

		final byte[] me3f={0x3f, 100, 0x00, 0x01, 0x69, 0x01};  //Barrido parcial de 0º a 180º de 100 promedios
		final byte[] me36={0x36, 100}; // 100 promedios
		final byte[] me37={0x37, 0x00, 0x01, 0x69, 0x01};  //Barrido parcial de 0º a 180º
		final byte[] me30={0x30,0x01}; //Datos de 1 barrido
		
		//A la vista de los parámetros elegimos el mensaje a mandar
		byte [] mensaje=null;
		if(numPromedios>1) {
			if(numPromedios>250)
				throw new LMSException("Numero de promedios >250");
			//Se epecifica promedio
			if (anguloInicial>0 || anguloFinal<180) {
				//se especifica promedio y águlos => mensaje 3f
				if( anguloInicial>anguloFinal || anguloFinal>180 || anguloInicial>180)
					throw new LMSException("Los angulos del barrido no están correctamente definidos");
				mensaje=me3f;
				mensaje[1]=(byte)numPromedios;
				UtilMensajes.word2Men((short)(anguloInicial*manMen.getResAngularCuartos()+1), mensaje, 2);  //nuevo angulo inicial
				UtilMensajes.word2Men((short)(anguloFinal*manMen.getResAngularCuartos()+1), mensaje, 4);  //nuevo angulo final				
			} else {
				//sólo promedio => mensaje 36
				mensaje=me36;
				mensaje[1]=(byte)numPromedios;
			}
		} else {
			//no se especifica promedio
			if (anguloInicial>0 || anguloFinal<180) {
				//se especifica algún águlo => mensaje 37
				if( anguloInicial>anguloFinal || anguloFinal>180 || anguloInicial>180)
					throw new LMSException("Los angulos del barrido no están correctamente definidos");
				mensaje=me37;
				UtilMensajes.word2Men((short)(anguloInicial*manMen.getResAngularCuartos()+1), mensaje, 1);  //nuevo angulo inicial
				UtilMensajes.word2Men((short)(anguloFinal*manMen.getResAngularCuartos()+1), mensaje, 3);  //nuevo angulo final
			} else {
				//no se especifica nada => Mensaje 30 al LSM
				mensaje=me30;
			}
		}
		manTel.EnviaMensaje(mensaje,50); //no nos preocupamos por la confirmación

		pidiendo=PIDIENDO_BARRIDO;
	}

	/**
	 * @return Barrido recibido
	 * @throws LMSException 
	 */
	public BarridoAngular recibeBarrido() throws LMSException {
		if(pidiendo!=PIDIENDO_BARRIDO)
			throw new LMSException("No acabamos de pedir barrido");
		//Recibimos respuesta
		byte[] respMen=manTel.LeeMensaje();
		pidiendo=PIDIENDO_NADA; //si falla termina la peticion
		if(respMen==null)
			throw  new LMSException("no se pudo leer mensaje de respuesta.");
		
		BarridoAngular barr=manMen.mensajeABarridoAngular(respMen);
		if(barr==null) 
			throw new LMSException("no se pudo interpretar correctamente mensaje.");
		return barr;
	}

	/**
	 * Solicita en envío continuo de barridos al LMS con los parámetros indicados
	 * @param anguloInicial en grados
	 * @param anguloFinal en grados
	 * @param numPromedios
	 * @throws LMSException si se hay algún problema
	 */
	public void pideBarridoContinuo(short anguloInicial, short anguloFinal,
			short numPromedios) throws LMSException {
		
		configuraInicial();
		if(pidiendo!=PIDIENDO_NADA && pidiendo!=PIDIENDO_CONTINUO 
				&& pidiendo!=PIDIENDO_PARAR)
			throw new LMSException("Estamos en medio de una peticion:" + pidiendo);
		

		final byte[] me28={0x20, 0x28, 100, 0x00, 0x01, 0x69, 0x01};  //Barrido parcial de 0º a 180º de 100 promedios
		final byte[] me26={0x20, 0x26, 100}; // 100 promedios
		final byte[] me27={0x20, 0x27, 0x00, 0x01, 0x69, 0x01};  //Barrido parcial de 0º a 180º
		final byte[] me24={0x20, 0x24}; //Datos de 1 barrido
		
		//A la vista de los parámetros elegimos el mensaje a mandar
		byte [] mensaje=null;
		if(numPromedios>1) {
			if(numPromedios>250)
				throw new LMSException("Numero de promedios >250");
			//Se epecifica promedio
			if (anguloInicial>0 || anguloFinal<180) {
				//se especifica promedio y águlos => mensaje 28
				if( anguloInicial>anguloFinal || anguloFinal>180 || anguloInicial>180)
					throw new LMSException("Los angulos del barrido no están correctamente definidos");
				mensaje=me28;
				mensaje[2]=(byte)numPromedios;
				UtilMensajes.word2Men((short)(anguloInicial*manMen.getResAngularCuartos()+1), mensaje, 3);  //nuevo angulo inicial
				UtilMensajes.word2Men((short)(anguloFinal*manMen.getResAngularCuartos()+1), mensaje, 5);  //nuevo angulo final				
			} else {
				//sólo promedio => mensaje 26
				mensaje=me26;
				mensaje[2]=(byte)numPromedios;
			}
		} else {
			//no se especifica promedio
			if (anguloInicial>0 || anguloFinal<180) {
				//se especifica algún águlo => mensaje 27
				if( anguloInicial>anguloFinal || anguloFinal>180 || anguloInicial>180)
					throw new LMSException("Los angulos del barrido no están correctamente definidos");
				mensaje=me27;
				UtilMensajes.word2Men((short)(anguloInicial*manMen.getResAngularCuartos()+1), mensaje, 2);  //nuevo angulo inicial
				UtilMensajes.word2Men((short)(anguloFinal*manMen.getResAngularCuartos()+1), mensaje, 4);  //nuevo angulo final
			} else {
				//no se especifica nada => Mensaje 24 al LSM
				mensaje=me24;
			}
		}

		System.out.println("Vamos a tratar de pedir envío continuo: "+UtilMensajes.hexaString(mensaje));
		boolean menOK=false;
		for(int ni=1; !menOK && ni<=numReintentos; ni++) {
			try { Thread.sleep(milisPreviosEnvio); } catch (InterruptedException e) {}
			manTel.purgaBufferEntrada();
			manTel.EnviaMensaje(mensaje); 
			byte[] respContinuo;
			if(( respContinuo=manTel.LeeMensaje())==null)
				continue; //no hay respuesta volvemos a intentarlo

			byte[] respModoContOK={(byte)0xa0, 00, 0x10};
			//está listo si se recibe confirmación o se empiezan a recibir barridos
			menOK=Arrays.equals(respContinuo,respModoContOK) ||
			   (manMen.mensajeABarridoAngular(respContinuo)!=null);
		}
		if(!menOK)
			throw new LMSException("Error al tratar de pasar al modo continuo :"+UtilMensajes.hexaString(mensaje));

		thContinuo.activar(); //ya esta activo, solo tenemos que activarlo
		
		pidiendo=PIDIENDO_CONTINUO;
		System.out.println("Se confirmó el envío continuo");
	}

	public void pidePararContinuo() throws LMSException {
		if(pidiendo!=PIDIENDO_CONTINUO && pidiendo!=PIDIENDO_PARAR)
			throw new LMSException("No estamos en medio de una peticion continua");
		System.out.println("Tratamos de parar envío continuo");
		byte[] menModo25={0x20, 0x25};
//		if(!
				manTel.EnviaMensajeSinConfirmacion(menModo25);
//				)
//			throw new LMSException("Error al enviar el mensaje "+UtilMensajes.hexaString(menModo25));
		//la confirmación la recibirá el thread continuo
		
		pidiendo=PIDIENDO_PARAR;
	}
	
	
	/**
	 * @return último barrido recibido de forma continua. No se tiene en cuenta
	 * el estado actual. Se puede pedir el último barrido después de terminado 
	 * el envío continuo 
	 */
	public BarridoAngular ultimoBarrido() {
		synchronized (mutexBarridos) {
			return ultimoBarrido;
		}
	}


	/** Para la recepción de los barridos continuos. 
	 * Se arranca suspendido. 
	 * Para que comience a recibir hay que {@link #activar()}.
	 * Cuando se recibe el mensaje de confirmación de parada se suspende automáticamente.
	 * @author verdino
	 *
	 */
	class ThreadContinuo extends Thread {
		
		private boolean suspendido=true;
		
		public synchronized void activar() {
			if(suspendido) {
				suspendido=false;
				notify();
			}
		}
		
		public synchronized  boolean isSuspendido() {
			return suspendido;
		}
		
		public void run() {
//			while(pidiendo==PIDIENDO_CONTINUO || pidiendo==PIDIENDO_PARAR) {
			while(true) { //se saldrá solo cuando se reciba confirmación mensaje parada
//				System.out.println("Esperamos mensaje");
				//miramos a ver si estamos suspendidos
				try {
					synchronized (this) {
						while (suspendido) wait(); 
						}
				} catch (InterruptedException e) {	}
				byte[] menBarridoC=manTel.LeeMensaje(0);
				if(menBarridoC==null) {
					System.err.println("no se ha recibido mensaje correctamente"); 
					//TODO llevar contador de fallos
					continue;  //no se ha recibido mensaje correctamente :-(
				}
				
				//vemos si se trata de mensaje de confirmación de cambio de modo
				if(pidiendo==PIDIENDO_PARAR 
						&& menBarridoC[0]==(byte)0xA0 
						&& menBarridoC[1]==0x00 
				) {
					System.out.println("Se confirmo la peticion de parada de barrido continuo. TERMINAMOS");
					pidiendo=PIDIENDO_NADA;
					suspendido=true;
				}
				//TODO optimizar para que se reutilice el barrido
				BarridoAngular barr=manMen.mensajeABarridoAngular(menBarridoC);
				if(barr==null) {
					System.err.println("no se ha podido convertir a barrdio :-(");
					//TODO llevar contador de fallos
					continue;  //no se ha podido convertir a barrdio :-(
				}
				//Tenemos un nuevo barrido valido lo ponemos en el vector
				synchronized (mutexBarridos) {
					//por ahora sólo apuntamos el nuevo barrido
					ultimoBarrido=barr;
				}
			}
		}
		
	}
	
}
