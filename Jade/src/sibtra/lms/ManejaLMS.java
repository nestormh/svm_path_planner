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

	private boolean configuradoCodigo;

	private boolean configuradoVariante;
	
	/**
	 * variable para saber en que estado estamos. Si estamos a la espera de alguna respuesta. 
	 */
	private int pidiendo;
	
	private static final int PIDIENDO_NADA=0;
	private static final int PIDIENDO_ZONA=0;
	private static final int PIDIENDO_BARRIDO=0;
	private static final int PIDIENDO_DISTNCIAS=0;
	
	/**
	 * Constructor, por defecto usa ManejaTelegramasJNI
	 * @param puertoSerie
	 */	
	public ManejaLMS(String puertoSerie){
		//valores por defecto supuestos
		manMen=new UtilMensajes((short)50,(short)180,true,(byte)0);

		//conectamos al puerto
		manTel=new ManejaTelegramasJNI();
		manTel.ConectaPuerto(puertoSerie);

		pidiendo=PIDIENDO_NADA;
		
		configuradoVariante=false;
		configuradoCodigo=false;
		
		try { configura(); } catch (LMSException e) {};
		
	}
	
	/**
	 * Constructor
	 * @param puertoSerie
	 * @param mantel objeto de manajo de telegramas a usar
	 */
	public ManejaLMS(String puertoSerie, ManejaTelegramas mantel) {
		manTel=mantel;
		manTel.ConectaPuerto(puertoSerie);
		
		//valores por defecto supuestos
		manMen=new UtilMensajes((short)50,(short)180,true,(byte)0);

		pidiendo=PIDIENDO_NADA;
		
		configuradoVariante=false;
		configuradoCodigo=false;
		
		try { configura(); } catch (LMSException e) {};
		
		
	}
	
	private void configura() throws LMSException {
		
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
		if(pidiendo!=PIDIENDO_NADA)
			throw new LMSException("Estamos en medio de una peticion");
		//pasamos al modo 25
		byte[] menModo25={0x20, 0x25};
		if(!manTel.EnviaMensaje(menModo25))
			throw new LMSException("Telegrama incorrecto en el cambio al modo 25: "
					+UtilMensajes.hexaString(menModo25));
		byte[] respModo25;
		if(( respModo25=manTel.LeeMensaje())==null)
			throw new LMSException("Respuesta incorrecta al cambio al modo 25");

		byte[] respModo25OK={(byte)0xa0, 00, 0x10};
		if(respModo25==null ||  !Arrays.equals(respModo25,respModo25OK))
			throw new LMSException("No fue posible el cambio al modo 25: "
					+UtilMensajes.hexaString(respModo25));
	}
	/**
	 * Manda los mensajes para cambiar a modo 00 (instalación)
	 * @throws LMSException  si hay problemas en la comunicación.
	 */
	protected void CambiaAModoInstalacion() throws LMSException {
		//pasamos al modo 00
		byte[] menModo00=new byte[LMSPassword.length()+2];
		menModo00[0]=0x20; menModo00[1]=0;
		for(int i=0; i<LMSPassword.length();i++)
			//la conversión debe funcionar ya que caracteres ASCII sólo usan 7 bits.
			menModo00[i+2]=(byte)LMSPassword.charAt(i);
		if(!manTel.EnviaMensaje(menModo00))
			throw new LMSException("Telegrama incorrecto en el cambio al modo instalacion: "
					+UtilMensajes.hexaString(menModo00));
		byte[] respModo00;
		if(( respModo00=manTel.LeeMensaje())==null)
			throw new LMSException("Respuesta incorrecta al cambio al modo instacion");

		byte[] respModo00OK={(byte)0xa0, 00, 0x10};
		if(respModo00==null ||  !Arrays.equals(respModo00,respModo00OK)) 
			throw new LMSException("No fue posible el cambio al modo instalación: "
					+UtilMensajes.hexaString(respModo00));
	}
	
	/**
	 * Obtiene la parte 1 de la configuración a través de mensaje 74
	 * @return Mensaje entero de respuesta f4, incuyendo byte de status.
	 * @throws LMSException  si hay problemas en la comunicación.
	 */
	protected byte[] ObtieneParte1Configuracion() throws LMSException {
		if(pidiendo!=PIDIENDO_NADA)
			throw new LMSException("Estamos en medio de una peticion");
		byte[] menMiraConfiguracion={0x74};
		if(!manTel.EnviaMensaje(menMiraConfiguracion))
			throw new LMSException("Telegrama incorrecto al mirar configuración: "
					+UtilMensajes.hexaString(menMiraConfiguracion));

		byte[] respConfig;
		if( (respConfig=manTel.LeeMensaje())==null  
				|| (respConfig.length!=36) || respConfig[0]!=(byte)0xf4 )
			throw new LMSException("Respuesta incorrecta a la consulta de configuración ");
		return respConfig;
	}
	
	/**
	 * Envía mensaje de cambio de configuración de parte 1
	 * @param menConfigura mensaje adecuado (77) para el cambi 
	 * @throws LMSException  si hay problemas en la comunicación.
	 */
	protected void CambiaParte1Configuracion(byte[] menConfigura) throws LMSException {
		if(pidiendo!=PIDIENDO_NADA)
			throw new LMSException("Estamos en medio de una peticion");
		if(!manTel.EnviaMensaje(menConfigura)) 
			throw new LMSException("Problema al enviar (77) la configuración");

		byte [] respNuevaConfig;
		if( (respNuevaConfig=manTel.LeeMensaje())==null
				|| !UtilMensajes.esConfirmacion(respNuevaConfig, menConfigura)
		) 
			throw new LMSException("No se aceptó la nueva configuración");
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

		final byte[] menVariante={(byte)0x3b, (byte)0xb4, 0, 0x32, 0};  //Mensaje ejemplo
		UtilMensajes.word2Men(rangoAngular,menVariante,1);  //valores actuales
		UtilMensajes.word2Men(resAngular, menVariante, 3);
		if(!manTel.EnviaMensaje(menVariante)) 
			throw new LMSException("Telegrama incorrecto al enviar cambio de variante: "
					+UtilMensajes.hexaString(menVariante));

		byte[] respVariante;
		if((respVariante=manTel.LeeMensaje())==null
				||  !UtilMensajes.esConfirmacion(respVariante,menVariante)
		) 
			throw new LMSException("No fue posible el cambio de variante: "
					+UtilMensajes.hexaString(respVariante));
		//Ponemos rango 180º y 1/2 grado de resolucion
		manMen.setRangoAngular(rangoAngular);
		manMen.setResAngular(resAngular); 
		configuradoVariante=true;
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
		configura();
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
		configura();
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
		configura();
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
		configura();
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
		configura();
		if(!configuradoCodigo)
			throw new LMSException("Codigo no configurado");
		return manMen.getCodigoRango();
	}
	
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
	 * @throws LMSException
	 */
	public void pideZona(byte queZona, boolean elConjunto1) throws LMSException {
		if(pidiendo!=PIDIENDO_NADA)
			throw new LMSException("Estamos en medio de una peticion");
		//A la vista de los parámetros elegimos el mensaje a mandar
		byte[] men45= {0x45, (byte)(elConjunto1?1:2), queZona};
		if(!manTel.EnviaMensaje(men45)) 
			throw new LMSException("Error al enviar el mensaje "+UtilMensajes.hexaString(men45));
		pidiendo=PIDIENDO_ZONA;
	}
	
	/** recibe la zona solicitada previamente con {@link #pideZona(byte, boolean)} */
	public ZonaLMS recibeZona() throws LMSException {
		if(pidiendo!=PIDIENDO_ZONA)
			throw new LMSException("No se acaba de pedir zona");
		//Recibimos respuesta
		byte[] respMen=manTel.LeeMensaje();
		if(respMen==null) 
			throw new LMSException("no se pudo leer mensaje de respuesta.");
		ZonaLMS zn=ZonaLMS.MensajeC5AZona(respMen);
		if(zn==null)
			throw new LMSException("Mensaje no se pudo convertir a zona");
		return zn;
	}
	
	/**
	 * Manda telegrama 30 solicitando las distancias verticales.
	 * @throws LMSException  si hay problemas en la comunicación.
	 */
	public void solicitaDistancia() throws LMSException {
		if(pidiendo!=PIDIENDO_NADA)
			throw new LMSException("Estamos en medio de una peticion");
		final byte[] men30={0x30,0x02}; //Datos de 1 barrido

		if(!manTel.EnviaMensaje(men30))
			throw new LMSException("Error al enviar el mensaje "+UtilMensajes.hexaString(men30));
		pidiendo=PIDIENDO_DISTNCIAS;
	}

	/**
	 * Recive del LMS el telegrama de las distancias vertical de las zonas
	 * @return distancia de cada zona en milímetros
	 * @throws LMSException si hay problemas en la comunicación.
	 */
	public double[] recibeDistancia() throws LMSException {
		if(pidiendo!=PIDIENDO_DISTNCIAS)
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
	
	/**
	 * 
	 */
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
		
		configura();
		if(pidiendo!=PIDIENDO_NADA)
			throw new LMSException("Estamos en medio de una peticion");

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
		if(!manTel.EnviaMensaje(mensaje))
			throw new LMSException("Error al enviar el mensaje "+UtilMensajes.hexaString(mensaje));

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
		if(respMen==null)
			throw  new LMSException("no se pudo leer mensaje de respuesta.");
		
		BarridoAngular barr=manMen.mensajeABarridoAngular(respMen);
		if(barr==null) 
			throw new LMSException("no se pudo interpretar correctamente mensaje.");

		return barr;
	}


}
