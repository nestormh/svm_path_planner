package sibtra.gps;

import gnu.io.SerialPortEvent;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import sibtra.imu.UtilMensajesIMU;
import sibtra.log.LoggerArrayInts;
import sibtra.log.LoggerDouble;
import sibtra.log.LoggerFactory;

/** Clase de conexión a GPS Triumph que maneja mensajes propietarios
 * Como:
 * 	,jps/PO,jps/ET
 * out,,jps/RT,jps/PO,jps/RD
 * 
 * em,,{jps/RT,nmea/GGA}:2
 * @author alberto
 *
 */

public class GPSConnectionTriumph extends GPSConnection {

	
	/** Tamaño máximo del mensaje */
	private static final int MAXMENSAJE = 5000;

	/** comando que se envía al GPS para comenzar el envío periódico de datos. Hay varias opciones
	 * <ul>
	 * <li>"em,,{jps/RT,nmea/GGA,jps/PO,jps/BL,nmea/GST,jps/DL,jps/ET}:0.2\n"
	 * <li>"%em%em,,{nmea/{GGA:0.2,GSA,GST,VTG},jps/DL}:1\n"
	 * <li>"%em%em,,{nmea/GGA:0.2,nmea/GSA,nmea/GST,nmea/VTG}:1\n"
	 * <li> (Actual) "%em%em,,{nmea/GGA:0.2,nmea/GSA,nmea/GST,nmea/VTG,jps/DL:0.4}:1\n"
	 * <li>"%em%em,,{jps/RT,nmea/GGA,jps/PG,jps/ET}:5\n"
	 * </ul>
	 */
	private static final String comandoPeriodicoPorDefecto = "%em%em,,{nmea/GGA:0.2,nmea/GSA,nmea/GST,nmea/VTG,jps/DL:0.4}:1\n";

//	private static final double NULLANG = -5 * Math.PI;


	/** Buffer en la que se van almacenando los trozos de mensajes que se van recibiendo */
	private byte buff[] = new byte[MAXMENSAJE];

	/** Indice inicial de un mensaje correcto */
	private int indIni=0;
	/** Indice final de un mensaje correcto */
	private int indFin=-1;
	
	/** largo del mensaje binario */
	private int largoMen;

	/** Banderín que indica que el mensaje es binario */
	private boolean esEstandar=false;

	/** Banderín que indica que el mensaje es de texto */
	private boolean esTexto=false;

	/** Calidad del enlace con la base. NaN si no se ha recibdo paquete DL */
	double calidadLink=Double.NaN;

	/** Indica estado de la depuracion */
	protected int nivelLog=0;

	private int numOKLink;

	/** Para registrar enteros que vienen en paquete DL */
	private LoggerArrayInts logDL;

	/** Para registrar calidad que viene en paquete DL */
	private LoggerDouble logCalDL;

	/** Ultimo comando de envío periódico que se envió. Permite la reactivación si es necesario dar un comando intermedio */
	private String comandoEnvioPeriodico=null;

	/** Para conseguir la exclusión mutua y el bloquo a la espera de respuesta de texto */
	private Object mutexRespuestaTexto=new Object();

	/** Contendrá la última respuesta de texto no considerada que se ha recibido */
	private String respuestaTexto=null;

	private GPSData ultimaPosicionBase;
	
	public static int ERR=0;
	public static int WAR=5;
	public static int INFO=15;
	
	protected void log(int nivel,String msg) {
		if(nivel==0) {
			System.err.println(msg);
		}
		if(nivel<=nivelLog) {
			System.out.println(msg);
		}
	}
	
	/**
	 * Constructor por defecto no hace nada.
	 * Para usar el puerto hay que invocar a {@link #setParameters(SerialParameters)} 
	 * y luego {@link #openConnection()}
	 */
	public GPSConnectionTriumph() {
		super();
	}

	/**
	 * Crea conexión a Triumph en puerto serial y 115200 baudios.
	 * Configura el envío periódico.
	 * @param portName nombre puerto donde encontrar al GPS
	 * @param baudio velocidad de la comunicacion en baudios
	 */
	public GPSConnectionTriumph(String portName) throws SerialConnectionException {
		this(portName,115200);
	}
	/**
	 * Crea conexión a Triumph en puerto serial y baudios indicados.
	 * Configura el envío periódico.
	 * @param portName nombre puerto donde encontrar al GPS
	 * @param baudio velocidad de la comunicacion en baudios
	 */
	public GPSConnectionTriumph(String portName, int baudios) throws SerialConnectionException {
		this(portName,baudios,ERR);
	}
	
	/**
	 * Crea conexión a Triumph en puerto serial y baudios indicados.
	 * Configura el envío periódico.
	 * @param portName nombre puerto donde encontrar al GPS
	 * @param baudio velocidad de la comunicacion en baudios
	 * @param nivelLog nivel para los mensajes de depuración ({@value #INFO}, {@value #WAR} ó {@value #ERR})
	 */
	public GPSConnectionTriumph(String portName, int baudios, int nivelLog) throws SerialConnectionException {
		super(portName,baudios);
		this.nivelLog=nivelLog;
		
		//ponemos muestras por segundo a la frecuencia de los GGA
		logLocales.setMuestrasSg(5);
		logEdadCor.setMuestrasSg(5);
		//TODO añadir loggers para todos los mensajes considerados.	
		logDL=LoggerFactory.nuevoLoggerArrayInts(this, "enterosDL",1);
		logDL.setDescripcion("Columnas [timeLast, numOK, numCorrup");
		logCalDL=LoggerFactory.nuevoLoggerDouble(this, "calidadDL",1);
	}


	/** Manda comando de envío periódico y lo apunta en {@link #comandoEnvioPeriodico} */
	public void comienzaEnvioPeriodico(String comandoEnvioPeriodico) {
		this.comandoEnvioPeriodico=comandoEnvioPeriodico;
		comandoGPS("%dM%dm\n");
		comandoGPS(comandoEnvioPeriodico);
	}
	
	/** Solicita la posición de la base varias veces hasta que la consiga o pasen los intentos
	 * @return si se consiguió posición de la base */
	public boolean esperaCentroBase(int intentos) {
		int intact=intentos;
		long espera=5000;
		while ((posicionDeLaBase(true)!=null) && (--intact)>0) { 
			try {
				Thread.sleep(espera); //esperamos
				espera*=2; //duplicamos la espera para la siguiente vez
			} catch (Exception e) {};
		}
		return ultimaPosicionBase!=null;
	}
	
	/** invoca {@link #esperaCentroBase(int)} con 10 intentos */
	public boolean esperaCentroBase() {
		return esperaCentroBase(10);
	}
	
	/** Manda úlimo comando de envío periódico en {@link #comandoEnvioPeriodico},
	 *  o el comando por defecto en {@link #comandoPeriodicoPorDefecto} si no se ha mandado uno previo */
	public void comienzaEnvioPeriodico() {
		if (comandoEnvioPeriodico==null)
			comandoEnvioPeriodico=comandoPeriodicoPorDefecto;
		comandoGPS("%dM%dm\n");
		comandoGPS(comandoEnvioPeriodico);
	}
	
	/** Obtiene la posición de la base */
	public GPSData posicionDeLaBase() {
		return posicionDeLaBase(false);
	}
	
	public GPSData posicionDeLaBase(boolean refrescada) {
		if( ultimaPosicionBase==null || refrescada) {
			//Tenemos que pedir la posición de la base
			String respuesta=null; //copia local de la respuesta
			String prefijo="%"+Thread.currentThread().getId()+"%";
			synchronized (mutexRespuestaTexto) {
				ultimaPosicionBase=null;
				respuestaTexto=null;
				comandoGPS(prefijo+"print,/par/pos/pd/ref/pos/geo\n");
				try {
					do {
					mutexRespuestaTexto.wait(5000);
					} while (respuestaTexto!=null && !respuestaTexto.startsWith(prefijo));
				} catch (InterruptedException e) {
					System.out.println(getClass().getName()+": Interrumpido esperando respuesta");
				}
				respuesta=respuestaTexto;
			}
			if(respuesta==null ) {
				System.err.println(getClass().getName()+": No se obtuvo respuesta o no corresponde:"+respuesta);
				return null;
			} else {
				try {
					//Tenemos la respuesta a nuestro comando, parseamos la posición de la base
					String[] campos=respuesta.substring(respuesta.indexOf('{')+1
							, respuesta.indexOf('}')-1).split(",");
					if(campos[0].equals("UNDEF"))
						//Rober no conoce la posición de la base
						return null;
					if(!campos[0].equals("W84")) {
						System.err.println(getClass().getName()+": la posición no es W84:"+respuesta);
						return null;
					} 
					String strLat=campos[1];
					//grados
					double latitud=Double.valueOf(strLat.substring(1, strLat.indexOf('d')));
					//minutos
					latitud+=Double.valueOf(strLat.substring(strLat.indexOf('d')+1,strLat.indexOf('m')))/60.0;
					//segundos
					latitud+=Double.valueOf(strLat.substring(strLat.indexOf('m')+1,strLat.indexOf('s')))/3600.0;
					//signo
					latitud*=(strLat.substring(0,1).equals("N"))?1.0:-1.0;

					String strLon=campos[2];
					//grados
					double longitud=Double.valueOf(strLon.substring(1, strLon.indexOf('d')));
					//minutos
					longitud+=Double.valueOf(strLon.substring(strLon.indexOf('d')+1,strLon.indexOf('m')))/60.0;
					//segundos
					longitud+=Double.valueOf(strLon.substring(strLon.indexOf('m')+1,strLon.indexOf('s')))/3600.0;
					//signo
					longitud*=(strLon.substring(0,1).equals("E"))?1.0:-1.0;

					double altura=Double.valueOf(campos[3]);

					ultimaPosicionBase=new GPSData(latitud,longitud,altura);
				} catch (Exception e) {
					System.err.println(getClass().getName()+": No se pudo interpretar correctamente la posición de la base"
							+ " con respuesta >"+respuesta+"<"
							+":"+e.getMessage());
					return null;
				}
			}
		}
		return ultimaPosicionBase;
	}
	

	/**
	 * Maneja los eventos seriales {@link SerialPortEvent#DATA_AVAILABLE}.
	 * Si se recibe un mensaje completo del GPS {@link #nuevaCadenaNMEA(String)}
	 */
	public synchronized void serialEvent(SerialPortEvent e) {
		if (e.getEventType() == SerialPortEvent.DATA_AVAILABLE) {
			try {
				while (inputStream.available() != 0) {
					int val = inputStream.read();
					//añadimos nuevo byte recibido
					if(indFin==(buff.length-1)) {
						log(ERR,"Buffer se llenó. Resetamos");
						indIni=0; indFin=-1; esEstandar=false; esTexto=false;
					}
					indFin++;
					buff[indFin]= (byte) val;
					if(esTexto) {
						if ( buff[indFin] == 10 || buff[indFin]==13)
						{
							//mensaje de texto completo
							indFin--; //quitamos caracter del salto
							if(indFin<=indIni) {
								indIni=0; indFin=-1; esEstandar=false; esTexto=false;
								continue;
							}
							//TODO da errores de String index out of range: -1
							String menTexto=new String(buff,indIni,(indFin-indIni+1));
							if(menTexto.charAt(0)=='$') {
								log(INFO,"Recibida NMEA:"+menTexto);
								cuentaPaquetesRecibidos++;
								nuevaCadenaNMEA(menTexto);
							}
							else {
								log(INFO,"Recibida Texto:"+menTexto);
								cuentaPaquetesRecibidos++;
								nuevaCadenaTexto(menTexto);
							}
							indIni=0; indFin=-1; esEstandar=false; esTexto=false;
						} 
					} else if (esEstandar) {
						//terminamos si ya está el tamaño
						if ( (indFin-indIni+1)==(largoMen+5) ) {
							//tenemos el mensaje estandar completo
//							System.out.print('e');
							cuentaPaquetesRecibidos++;
							nuevaCadenaEstandar();						
							indIni=0; indFin=-1; esEstandar=false; esTexto=false;
						}
					} else { //Todavía no sabemos si es texto o binario
						boolean sincronizado=false;
						while(!sincronizado && (indFin>=indIni)) {
							int larAct=indFin-indIni+1;
							larAct=indFin-indIni+1;
							if (larAct==1) {
								if (isCabTex(indIni)) {
									esTexto=true;
									sincronizado=true;
								}
								else 
									if (!isCabBin(indIni)) {
									//no es cabecera permitida, nos resincronizamos
									indIni=0; indFin=-1; //reiniciamos el buffer
								} else //por ahora puede ser binaria
									sincronizado=true;
								continue;
							} 
							if (larAct==2) {
								if  (!isCabBin(indIni) || !isCabBin(indIni+1))  {
									//no es cabecera permitida, nos resincronizamos
									indIni++;
								} else //por ahora puede ser binaria 
									sincronizado=true;
								continue;
							}
							if (larAct==3) {
								if (!isCabBin(indIni) || !isCabBin(indIni+1) || !isHexa(indIni+2)) {
									//no es caracter hexa, nos resincronizamos
									indIni++;
								} else  //por ahora puede ser binaria 
									sincronizado=true;
								continue;
							}
							if (larAct==4) {
								if (!isCabBin(indIni) || !isCabBin(indIni+1) 
										|| !isHexa(indIni+2) || !isHexa(indIni+3)) {
									//no es caracter hexa, nos resincronizamos
									indIni++;
								} else  //por ahora puede ser binaria 
									sincronizado=true;
								continue;
							} 
							//caso de largo 5
							if (!isCabBin(indIni) || !isCabBin(indIni+1) 
									|| !isHexa(indIni+2) || !isHexa(indIni+3) || !isHexa(indIni+4)) {
								//no es caracter hexa, nos resincronizamos
								indIni++;
							} else { //estamos seguros que es binaria 
								sincronizado=true;
								esEstandar=true;
								largoMen=largo();
							}

						}
					}
				}
			} catch (IOException ioe) {
				log(ERR,"\nError al recibir los datos");
			} catch (Exception ex) {
				log(ERR,"\nGPSConnectionTriump Error al procesar >"+buff+"< : " + ex.getMessage());
				ex.printStackTrace();
				indIni=-1;
			}
		}
	}
	
	/** @return el largo del mensaje a partir de los 3 caracteres exadecimales */ 
	private int largo() {
		int lar=0;
		int indAct=indIni+2;
		if(buff[indAct]<=(byte)'9') lar+=32*(buff[indAct]-(byte)'0');
		else lar+=32*(buff[indAct]-(byte)'A'+10);
		indAct++;
		if(buff[indAct]<=(byte)'9') lar+=16*(buff[indAct]-(byte)'0');
		else lar+=16*(buff[indAct]-(byte)'A'+10);
		indAct++;
		if(buff[indAct]<=(byte)'9') lar+=(buff[indAct]-(byte)'0');
		else lar+=(buff[indAct]-(byte)'A'+10);
		return lar;
	}

	/**
	 * Mira si es caracter hexadecimal MAYÚSCULA
	 * @param ind indice en {@link #buff} del dato a mirar
	 * @return true si es caracter hexadecimal mayúscula
	 */
	private boolean isHexa(int ind) {
		return ((buff[ind]>=(byte)'0') && (buff[ind]<=(byte)'9')) || ((buff[ind]>=(byte)'A' && buff[ind]<=(byte)'F')) ;
	}

	/**
	 * Mira si es caracter válido en la cabecera de paquete binario (entre 48 y 126)
	 * @param ind indice en {@link #buff} del dato a mirar
	 * @return true si es caracter es válido
	 */
	private boolean isCabBin(int ind) {
		return ((buff[ind]>=48) && (buff[ind]<=126)) ;
	}

	/**
	 * Mira si es caracter válido en la cabecera de paquete texto (entre 33 y 47)
	 * @param ind indice en {@link #buff} del dato a mirar
	 * @return true si es caracter es válido
	 */
	private boolean isCabTex(int ind) {
		return ((buff[ind]>=33) && (buff[ind]<=47)) ;
	}

	/** Se invoca cuando se recibe una cadena de texto que no es NMEA 
	 * Se apunta en {@link #respuestaTexto} y se avisa a los bloqueados en {@link #mutexRespuestaTexto} 
	 */
	void nuevaCadenaTexto(String mensaje) {
		log(INFO,"Recibida cadena de texto >"+mensaje+"<");
	}
	
	/** Se invoca cuando se recibe una cadena binaria propietaria GREIS */
	void nuevaCadenaEstandar() {
		int larMen=indFin-indIni+1;
		//iterpretamos los mensajes
		/* RT [~~]=126
		 * struct RcvTime {5} {
  			u4 tod; // Tr modulo 1 day (86400000 ms) [ms]
  			u1 cs; // Checksum
			};
		 */
		if(buff[indIni]==(byte)'~' && buff[indIni+1]==(byte)'~') {
			if(larMen!=(5+5)) {
				log(WAR,"El mensaje RT no tienen el tamaño correcto "+larMen+" Ignoramos mensaje");
				return;				
			}
			//comprobamos checksum
			byte csc=(byte)checksum8(buff, indIni, larMen-1);
			if(csc!=buff[indFin]) {
				log(WAR,"Error checksum "+csc+"!="+buff[indFin]+" Ignoramos mensaje");
				return;
			}
			//el checksum es correcto
			ByteBuffer bb = ByteBuffer.allocate(4);
			bb.order(ByteOrder.LITTLE_ENDIAN);
			bb.put(buff, indIni+5, 4);
			bb.rewind();
			int tod=bb.getInt();
			log(INFO,"\n\n\nMensaje RT: tod="+tod);
			return;
		}
		/* [::](ET) Epoch Time5 
   			struct EpochTime {5} {
     			u4 tod; // Tr modulo 1 day (86400000 ms) [ms]
     			u1 cs; // Checksum
   			};
		 */
		if(buff[indIni]==(byte)':' && buff[indIni+1]==(byte)':') {
			if(larMen!=(5+5)) {
				log(WAR,"El mensaje ET no tienen el tamaño correcto "+larMen+" Ignoramos mensaje");
				return;				
			}
			//comprobamos checksum
			byte csc=(byte)checksum8(buff, indIni, larMen-1);
			if(csc!=buff[indFin]) {
				log(WAR,"Error checksum "+csc+"!="+buff[indFin]+" Ignoramos mensaje");
				return;
			}
			//el checksum es correcto
			ByteBuffer bb = ByteBuffer.allocate(4);
			bb.put(buff, indIni+5, 4);
			bb.order(ByteOrder.LITTLE_ENDIAN);
			int tod=bb.getInt(0);
			log(INFO,"Mensaje ET: tod="+tod);
			return;
		}
		
		/* [PO] Cartesian Position
  			struct Pos {30} {
    			f8 x, y, z; //  Cartesian coordinates [m]
                    			Position SEP6 [m]
    		f4 sigma;    //
    		u1 solType; //  Solution type
    		u1 cs;       // Checksum
  			};
		 */
		if(buff[indIni]==(byte)'P' && buff[indIni+1]==(byte)'O') {
			if(larMen!=(5+30)) {
				log(WAR,"El mensaje PO no tienen el tamaño correcto "+larMen+" Ignoramos mensaje");
				return;				
			}
			//comprobamos checksum
			byte csc=(byte)checksum8(buff, indIni, larMen-1);
			if(csc!=buff[indFin]) {
				log(WAR,"Error checksum "+csc+"!="+buff[indFin]+" Ignoramos mensaje");
				return;
			}
			//el checksum es correcto
			ByteBuffer bb = ByteBuffer.allocate(30);
			bb.put(buff, indIni+5, 30);
			bb.rewind();
			bb.order(ByteOrder.LITTLE_ENDIAN);
			double x=bb.getDouble();
			double y=bb.getDouble();
			double z=bb.getDouble();
			float sigma=bb.getFloat();
			byte solType=bb.get();
			
			log(INFO,"Mensaje PO: ("+x+","+y+","+z+") sigma="+sigma+" solType="+solType);
			return;
		}

		/*
		 * [PG] Geodetic Position
             struct GeoPos {30} {
               f8 lat;      // Latitude [rad]
               f8 lon;      // Longitude [rad]
               f8 alt;      // Ellipsoidal height [m]
               f4 pSigma; // Position SEP [m]
               u1 solType; // Solution type
               u1 cs;       // Checksum
             };
		 */
		if(buff[indIni]==(byte)'P' && buff[indIni+1]==(byte)'G') {
			if(larMen!=(5+30)) {
				log(WAR,"El mensaje PG no tienen el tamaño correcto "+larMen+" Ignoramos mensaje");
				return;				
			}
			//comprobamos checksum
			byte csc=(byte)checksum8(buff, indIni, larMen-1);
			if(csc!=buff[indFin]) {
				log(WAR,"Error checksum PG "+csc+"!="+buff[indFin]+" Ignoramos mensaje");
				return;
			}
			//el checksum es correcto
			ByteBuffer bb = ByteBuffer.allocate(30);
			bb.put(buff, indIni+5, 30);
			bb.rewind();
			bb.order(ByteOrder.LITTLE_ENDIAN);
			double lat=bb.getDouble();
			double lon=bb.getDouble();
			double alt=bb.getDouble();
			float sigma=bb.getFloat();
			byte solType=bb.get();
			
			log(INFO,"Mensaje PG: ("+lat+","+lon+","+alt+") sigma="+sigma+" solType="+solType);
			return;
		}

		
		/* [BL] Base Line
  			struct BaseLine {34} {
    			f8 x, y, z; // Calculated baseline vector coordinates [m]
    			f4 sigma;    // Baseline Spherical Error Probable (SEP) [m]
    			u1 solType; // Solution type
    			i4 time;     // receiver time of the baseline estimate [s]
    			u1 cs;       // Checksum
  			};
		 */
		if(buff[indIni]==(byte)'B' && buff[indIni+1]==(byte)'L') {
			if(larMen!=(5+34)) {
				log(WAR,"El mensaje BL no tienen el tamaño correcto "+larMen+" Ignoramos mensaje");
				return;				
			}
			//comprobamos checksum
			byte csc=(byte)checksum8(buff, indIni, larMen-1);
			if(csc!=buff[indFin]) {
				log(WAR,"Error checksum "+csc+"!="+buff[indFin]+" Ignoramos mensaje");
				return;
			}
			//el checksum es correcto
			ByteBuffer bb = ByteBuffer.allocate(34);
			bb.put(buff, indIni+5, 34);
			bb.order(ByteOrder.LITTLE_ENDIAN);
			bb.rewind();
			double x=bb.getDouble();
			double y=bb.getDouble();
			double z=bb.getDouble();
			float sigma=bb.getFloat();
			byte solType=bb.get();
			int time=bb.getInt();

			log(INFO,"Mensaje BL: ("+x+","+y+","+z+") sigma="+sigma+" solType="+solType
					+" time="+time);
			return;
		}
		
		//Mensaje de texto DL
//		[DL] Data Link Status
//		This message displays the status of the data links associated with the corresponding
//		serial ports/modem.
//		        #       Format                                 Description
//		      1      DLINK       Title of the message
//		      2      %1D         Total number of data links (0…5). The rest of the message is avail-
//		                         able only if this number is non-zero. Otherwise the total number of
//		                         data links value is immediately followed by the checksum.
//		      3      ({%C,%C,%S, Group of fields associated with the given data link (note that the
//		             %3D,%4D,%4  total number of such groups is determined by the previous field).
//		             D,%.2F})    These fields are
//		                         - Data link identifier (refer to Table 4-8 below).
//		                         - Decoder identifier (“R” – RTCM decoder, “T” – RTCM 3.0
//		                         decoder, “C” – CMR decoder, “J” – JPS decoder).
//		                         - Reference station identifier.
//		                         - Time [in seconds] elapsed since receiving last message (maxi-
//		                         mum value = 999). Estimated with an accuracy of ±1 second.
//		                         - Number of received messages (between 0001 and 9999). If no
//		                         message has been received, this data field contains zero.
//		                         - Number of corrupt messages (between 0001 and 9999). If no cor-
//		                         rupt messages have been detected, this data field is set to zero.
//		                         - Data link quality in percent (0-100);
//		      4      @%2X        Checksum
//        Table 4-8. Data Link Identifiers
//        Id           Corresponding Stream
//    ‘A’…’D’ serial ports A…D, /dev/ser/a…/dev/ser/d
//    ‘E’…’I’  TCP ports A…E, /dev/tcp/a…/dev/tcp/d
//    ‘P’      TCP client port, /dev/tcpcl/a
//    ‘U’      USB port A, /dev/usb/a
//    ‘L’      Bluetooth port A, /dev/blt/a
//    ‘g’      CAN port A, /dev/can/a
//		
//		EJemplo:DL02B,DLINK,1,{D,C,0000,999,0000,0000,100.00}@62
//                VEMOS QUE FALTA LA COMA ANTES DE LA ARROBA^


		if(buff[indIni]==(byte)'D' && buff[indIni+1]==(byte)'L') {
			//no tiene un largo estandar pero si uno mínimo
			if(larMen<(5+9)) {
				log(WAR,"El mensaje DL no tienen el tamaño necesario "+larMen+". Ignoramos mensaje");
				return;				
			}

			//convertimos a string
			try {
				String cadena=new String(buff,indIni,larMen);
				int posArroba=cadena.indexOf('@');
				String CS=cadena.substring(posArroba+1);
				//TODO comprobamos checksum
				int csc=checksum8(buff, indIni, larMen-2);
				if( csc!=Byte.valueOf(CS,16) ) {
					log(WAR,"Error checksum "+csc+"!="+CS+". Ignoramos mensaje");
					return;
				}
				String[] campos=cadena.substring(0, posArroba).split(",");
				//el checksum es correcto
				if(campos.length<4) {
					log(WAR,"DL no tiene los campos mínimos necesarios. Ignoramos");
					return;
				}
				if(!campos[1].equals("DLINK")) {
					log(WAR,"DL no tiene título DLINK. Ignoramos");
					return;
				}
				int la=Integer.valueOf(campos[2]);
				int ca=3; //campo donde empezamos la búsqueda de {
				while(la>0) {
					while(campos[ca].charAt(0)!='{') ca++;
					char tipo=campos[ca++].charAt(1);
					char decoId=campos[ca++].charAt(0);
					String stationID=campos[ca++];
					int timeLast=Integer.valueOf(campos[ca++]);
					int numOK=Integer.valueOf(campos[ca++]);
					int numCorrup=Integer.valueOf(campos[ca++]);
					//TODO posible problema con }
					double quality=Double.valueOf(campos[ca].substring(0,campos[ca].lastIndexOf('}')));
					ca++;
					log(INFO,String.format("DL: %c %c %s %d %d %d %f"
							,decoId, tipo, stationID, timeLast, numOK, numCorrup,quality ));
					if(tipo=='D') { //puerto D es el del enlace
						calidadLink=quality;
						numOKLink=numOK;
						//apuntamos en loggers
						logDL.add(timeLast, numOK, numCorrup);
						logCalDL.add(quality);
					}
					la--;
				}
			} catch (Exception e) {
				log(WAR,"Error parseando campo DL:"+e.getMessage()+" Ignoramos");
			}
			
			return;
		}

		// Respuesta a un comando enviado
//		[RE] Reply
//		  struct RE {var} {
//		    a1 reply[]; // Reply
//		  };
		if(buff[indIni]==(byte)'R' && buff[indIni+1]==(byte)'E') {
			//convertimos a string
			try {
				//Quitamos RE y tamaño ###
				String cadena=new String(buff,indIni+5,larMen-5);
				log(INFO,"Respuesta: >"+cadena+"<");
				//No tiene Checksum que comprobar
				nuevaRespuesta(cadena);
			} catch (Exception e) {
				log(WAR,"Error parseando campo RE:"+e.getMessage()+" Ignoramos");
			}
			
			return;
		}
		
		//contenido del mensaje en crudo
		System.out.print("Binaria ("+larMen+"):"+new String(buff,indIni,5)+" >");
		for(int i=indIni+5; i<=indFin; i++)
			if (buff[i]<32 || buff[i]>126)
				//no imprimible
				System.out.print('.');
			else
				System.out.print((char)buff[i]);
		log(INFO,"< >"+UtilMensajesIMU.hexaString(buff, indIni+5, larMen-5)+"<");
	}

	/**
	 * Calcula checksum de 8 bits según se indica pag 363 de la 'GREIS Reference Guide'
	 * <code> 
typedef unsigned char u1;
enum {
  bits = 8,
  lShift = 2,
  rShift = bits - lShift
};
#define ROT_LEFT(val) ((val << lShift) | (val >> rShift))
u1 cs(u1 const* src, int count)
{
  u1 res = 0;
  while(count--)
    res = ROT_LEFT(res) ^ *src++;
  return ROT_LEFT(res);
}
</code>
@return valor calculado como int para no tener problema con el signo
 */
	public static int checksum8(byte[] buff, int ini, int largo) {
		int res=0;
		for(int i=ini; i<ini+largo; i++)
			res= (((res<<2)|(res>>>6)) ^ buff[i]) & 0xff; //nos aseguramos parte alta de  int a 0
		return ((res<<2)|(res>>>6))& 0xff;
	}
	
	/**
	 * @param comando comando de texto a enviar al GPS
	 */
	public void comandoGPS(String comando) {
		if(!isOpen()) {
			System.err.println(getClass().getName()+": Tratando de enviar comando cuando no tenemos conexion");
			return;
		}
		try {
		outputStream.write(comando.getBytes());
		log(INFO,"Enviado Comando:>"+comando+"<");
		} catch (Exception e) {
			log(WAR,"Problema al enviar comando Triumph:"+e.getMessage());
		}
	}

	/** Se invoca cada vez que se recive cadena RE###.
	 * pone {@link #respuestaTexto} con respuesta recibida y despierta thread esperando en {@link #mutexRespuestaTexto}
	 * @param mensaje el recibido (sin RE###)
	 */
	private void nuevaRespuesta(String mensaje) {
		synchronized (mutexRespuestaTexto) {
			respuestaTexto=mensaje;
			mutexRespuestaTexto.notifyAll();
		}
	}
	
	/** Calidad del enlace con la base */
	public double getCalidadLink() {
		return calidadLink;
	}
	
	/**
	 * @return numero de paquetes OK del Link
	 */
	public int getNumOKLink() {
		return numOKLink;
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {

		GPSConnectionTriumph gpsC;
				
		try {
			gpsC=new GPSConnectionTriumph("/dev/ttyUSB0",115200,INFO);
			if(gpsC.esperaCentroBase())
				gpsC.fijaCentro(gpsC.posicionDeLaBase());
			gpsC.comienzaEnvioPeriodico();
//			gpsC.comandoGPS("%Dm%dm\n");
			gpsC.comandoGPS("%DL%out,,jps/DL\n");

			try { Thread.sleep(5000); } catch (Exception e) {}
			
			gpsC.comandoGPS("%pb%print,/par/pos/pd/ref/pos/geo\n");
			gpsC.comandoGPS("%ver% print,rcv/ver\n");

			try { Thread.sleep(5000); } catch (Exception e) {}
//			gpsC.comandoGPS("%Dm%dm\n");
			GPSData posB=gpsC.posicionDeLaBase();
			System.out.println("Posición de la base:"+posB);
//
//			gpsC.comandoGPS("em,,{jps/RT,nmea/GGA,jps/PO,jps/BL,nmea/GST,jps/ET}:10\n");
			try { Thread.sleep(10000000); } catch (Exception e) {}
//
//			gpsC.comandoGPS("dm\n");
		} catch (Exception e) {
			System.err.println("Problema al usar el GPS:"+e.getMessage());
		}
		
		System.exit(0);
	}
}