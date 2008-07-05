/**
 * 
 */
package sibtra.imu;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.TooManyListenersException;

import gnu.io.CommPortIdentifier;
import gnu.io.NoSuchPortException;
import gnu.io.PortInUseException;
import gnu.io.SerialPort;
import gnu.io.SerialPortEvent;
import gnu.io.SerialPortEventListener;
import gnu.io.UnsupportedCommOperationException;

/**
 * @author alberto
 *
 */
public class ConexionSerialIMU implements SerialPortEventListener {

	
	private static final int MaxLen = 812;
	private CommPortIdentifier idPuertoCom;
	private SerialPort puertoSerie;
	InputStream flujoEntrada;
	OutputStream flujoSalida;
	private byte[] buf;
	private boolean abierto;
	private boolean enMensaje;
	private int indBuf;
	private int lenTot;

	/** Donde se almacena los últimos angulos recibidos */
	private AngulosIMU angulo;

	/**
	 * Inicialización del puerto serie. 
	 * @param NombrePuerto
	 * @return
	 */
	public boolean ConectaPuerto(String NombrePuerto) {
		return ConectaPuerto(NombrePuerto,5);
	}
	
	
	/**
	 * Inicialización del puerto serie. 
	 * @param NombrePuerto
	 * @param frecuencia
	 * @return
	 */
	public boolean ConectaPuerto(String NombrePuerto, double frecuencia) {
		try {
			idPuertoCom=CommPortIdentifier.getPortIdentifier(NombrePuerto);
		} catch (NoSuchPortException e) {
			System.err.println("\n Puerto no encontrado: "+NombrePuerto);
			return false;
		}

		//Obtenemos puerto serie
		try {
			puertoSerie = (SerialPort) idPuertoCom.open("TrataTelegramas", 200000);
		} catch (PortInUseException e) {
			System.err.println("\n Puerto "+NombrePuerto+" ya en uso por: "+e.currentOwner);
			return false;			
		}

		//Parámetros comunicación del puerto serie
		try {
			puertoSerie.setSerialPortParams(115200, SerialPort.DATABITS_8, 
					SerialPort.STOPBITS_1, 
					SerialPort.PARITY_NONE);
		} catch (UnsupportedCommOperationException e) {
			System.err.println("\n No puedo fijar los parámetros a puerto "+NombrePuerto);
			return false;			
		}

		//Fijamos TimeOut
		try {
			puertoSerie.enableReceiveTimeout(10*1000); //respuesta más lenta 10 sg
			if(puertoSerie.isReceiveTimeoutEnabled()) {
				System.err.println("\n Fijado timeout a : "+puertoSerie.getReceiveTimeout());
			} else {
				System.err.println("\n No se ha podido fijar el timeOut");
			}
		}  catch (UnsupportedCommOperationException e) {
			System.err.println("\n Puerto no soporta fijar TimeOut: "+e.getMessage());
			//return false; seguimos aunque no haya timeout			
		}

		//Tratamos de fijar buffer entrada
		puertoSerie.setInputBufferSize(15); //maxima longitud telegrama
		System.err.println("\nFijado buffer de entrada a :"+puertoSerie.getInputBufferSize());


//		//Fijamos el umbral de recepción
//		try {
//			puertoSerie.enableReceiveThreshold(15); //un byte se puede recibir => detectar confirmaciones
//			if(puertoSerie.isReceiveThresholdEnabled()) {
//				System.err.println("\n Fijado umbral a : "+puertoSerie.getReceiveThreshold());
//			} else {
//				System.err.println("\n No se ha podido fijar el umbral de entrada");
//			}
//		}  catch (UnsupportedCommOperationException e) {
//			System.err.println("\n Puerto no soporta fijar Umbral de entrada: "+e.getMessage());
//			//return false; seguimos aunque no haya 			
//		}

		
		
		try {
			flujoEntrada = puertoSerie.getInputStream();
		} catch (IOException e) {
			System.err.println("\n No se pudo obtener flujo de entrada para puerto "+NombrePuerto);
			return false;			
		}

		try {
			flujoSalida = puertoSerie.getOutputStream();
		} catch (IOException e) {
			System.err.println("\n No se pudo obtener flujo de salida para puerto "+NombrePuerto);
			return false;			
		}
		// Set notifyOnDataAvailable to true to allow event driven input.
		puertoSerie.notifyOnDataAvailable(true);
		try {
			puertoSerie.addEventListener(this);
		} catch (TooManyListenersException e) {
			System.err.println("Demasiados menajadores !!! "+e.getMessage());
		}
		
		abierto=true;
		buf=new byte[MaxLen]; // creamos del tamaño máximo de telegrama
		enMensaje=false;
		return initIMU() && fijaFrecuencia(frecuencia);
		
	}

	public boolean initIMU() {
		byte[] men;

		try {
			try{ Thread.sleep(500); } catch (InterruptedException e) {};
			//pasamos modo configuración
			byte[] menC={(byte)0xfa, (byte)0xff, 0x30, 0 , 0};
			men=menC;
			UtilMensajesIMU.fijaCRC(men);
			System.out.println("Enviamos mensaje "+UtilMensajesIMU.hexaString(men));
			flujoSalida.write(men);
			flujoSalida.flush();
			
			try{ Thread.sleep(500); } catch (InterruptedException e) {};
		
			//indicamos envíe  sólo orientacion
			byte[] menS={(byte)0xfa, (byte)0xff, (byte)0xd0, 2 , 0, 4, 0 };
			men=menS;
			UtilMensajesIMU.fijaCRC(men);
			System.out.println("Enviamos mensaje "+UtilMensajesIMU.hexaString(men));
			flujoSalida.write(men);
			flujoSalida.flush();
			
			try{ Thread.sleep(500); } catch (InterruptedException e) {};
			
			//indicamos envíe angulo y time stamp
			byte[] menA={(byte)0xfa, (byte)0xff, (byte)0xd2, 4 , 0,  0, 0, 5, 0 };
			men=menA;
			UtilMensajesIMU.fijaCRC(men);
			System.out.println("Enviamos mensaje "+UtilMensajesIMU.hexaString(men));
			flujoSalida.write(men);
			flujoSalida.flush();
			
			try{ Thread.sleep(500); } catch (InterruptedException e) {};
			//Volvemos a modo datos
			byte[] menD={(byte)0xfa, (byte)0xff, (byte)0x10, 0, 0 };
			men=menD;
			UtilMensajesIMU.fijaCRC(men);
			System.out.println("Enviamos mensaje "+UtilMensajesIMU.hexaString(men));
			flujoSalida.write(men);
			flujoSalida.flush();
			return true;
		}
		catch (IOException e) {
			System.err.println("Problema en inicialización al enviar mensaje:"+e.getMessage());
			return false;
		}


	}

	/** Cierra el puerto y los flujos de entrada y salida */
	public boolean cierraPuerto() {
		if(!abierto)
			return false;
		if(puertoSerie!=null)
			try {
				flujoEntrada.close();
				flujoSalida.close();
			} catch (IOException e) {
				System.err.println(e);
			}
		puertoSerie.close();
		abierto=false;
		return true;
	}

	
	
	/* (non-Javadoc)
	 * @see gnu.io.SerialPortEventListener#serialEvent(gnu.io.SerialPortEvent)
	 */
	public void serialEvent(SerialPortEvent e) {
		if (e.getEventType() == SerialPortEvent.DATA_AVAILABLE) {
			try {
				while (flujoEntrada.available() != 0) {
					int val = flujoEntrada.read();
					if (!enMensaje) { 
						if (val == 0xFA ) {
							lenTot=0;
							indBuf=0;
							buf[indBuf]=(byte)val;
							indBuf++;
							enMensaje=true;
						} 
						//caso contrario no hacemos nada => ignoramos caracteres
					} 
					else { //estamos en mensaje
						buf[indBuf]=(byte)val;
						indBuf++;
						if(indBuf==4) {
							lenTot=(((int)buf[3]&0xff)+5);
//							System.out.println("Mensaje longitud "+lenTot);
						}
						if(lenTot!=0 && indBuf==lenTot) {
							//Tenemos todo el mensaje
//							System.out.println("Recibido mensaje "+UtilMensajesIMU.hexaString(buf, 0, indBuf));
							if(UtilMensajesIMU.correctoCRC(buf, indBuf)) {
								//mensaje terminado y correcto
								enMensaje=false;
								procesaMensaje();
							} else  {
								System.err.println("Error en Checksum de mensaje");
							}
							enMensaje=false; //comenzamos el nuevo mensaje
						}
					}
				}
			} catch (IOException ioe) {
				System.err.println("\nError al recibir los datos");
//			} catch (Exception ex) {
//				System.err.println("\nCadena fragmentada : " + ex.getMessage());
				enMensaje=false;
			}
		}
		
	}

	/**
	 */
	private void procesaMensaje() {
		//System.out.println("Recibido mensaje "+UtilMensajesIMU.hexaString(buf, 0, indBuf));
		int lenData=(int)buf[3]&0xff;
		int iniData=4;
		if(lenData==255) {
			lenData=UtilMensajesIMU.men2Word(buf, 4);
			iniData=6;
		}
		switch(buf[2]) {
		case 0x32: 
			//MTData
			if(lenData==14) { //es tamaño correcto Angulos Euler + Contador
				try {
					//leemos los datos
					DataInputStream dis=new DataInputStream(new ByteArrayInputStream(buf,iniData,lenData));
					angulo=new AngulosIMU(dis.readFloat(),dis.readFloat(),dis.readFloat(),dis.readUnsignedShort());
//					System.out.printf("%7d: %15f %15f %15f\n",angulo.contador,angulo.roll,angulo.pitch,angulo.yaw);
					avisaListeners();
				} catch (IOException e) {
					System.err.println("Problemas al leer floats del mensaje");
				}
			} else {
				System.err.println("Tamaño de MTData no soportado: "+lenData+ " mensaje"
						+UtilMensajesIMU.hexaString(buf, 0, indBuf));
			}

			break;
		default:
			System.err.println("Mensaje de MID no considerado "+buf[2]+":"+UtilMensajesIMU.hexaString(buf, 0, indBuf));
		}
		return;

	}
	
	/**
	 * Trata de fijar la frecuencia de envío de datos tocando OutputSkipfactor
	 * @param herzios deseados en rango 100 a 1.53e-6
	 * @return true si se pudo, false si falló.
	 */
	public boolean fijaFrecuencia(double herzios) {
		if(herzios>100 || herzios<1.53e-6)
			return false;
		int skip=(int)Math.floor(100/herzios)-1;
		System.out.println("Usando skip="+skip);
		byte[] men;
		
		try {
			try{ Thread.sleep(500); } catch (InterruptedException e) {};

			//pasamos modo configuración
			byte[] menC={(byte)0xfa, (byte)0xff, 0x30, 0 , 0};
			men=menC;
			UtilMensajesIMU.fijaCRC(men);
			System.out.println("Enviamos mensaje "+UtilMensajesIMU.hexaString(men));
			flujoSalida.write(men);
			flujoSalida.flush();
			
			try{ Thread.sleep(500); } catch (InterruptedException e) {};
		
			//indicamos skipFacto
			byte[] menS={(byte)0xfa, (byte)0xff, (byte)0xd4, 2, 0, 0, 0 };
			menS[4]=(byte)((skip&0xff00)>>8);
			menS[5]=(byte)(skip&0xff);
			men=menS;
			UtilMensajesIMU.fijaCRC(men);
			System.out.println("Enviamos mensaje "+UtilMensajesIMU.hexaString(men));
			flujoSalida.write(men);
			flujoSalida.flush();
			
			try{ Thread.sleep(500); } catch (InterruptedException e) {};
			//Volvemos a modo datos
			byte[] menD={(byte)0xfa, (byte)0xff, (byte)0x10, 0, 0 };
			men=menD;
			UtilMensajesIMU.fijaCRC(men);
			System.out.println("Enviamos mensaje "+UtilMensajesIMU.hexaString(men));
			flujoSalida.write(men);
			flujoSalida.flush();
		}
		catch (IOException e) {
			System.err.println("Problema al enviar mensaje:"+e.getMessage());
			return false;
		}

		
		
		return true;
	}

	/** mantiene la lista de listeners */
	private ArrayList<IMUEventListener> listeners = new ArrayList<IMUEventListener>();

	/**
	 * Para añadir objeto a la lista de {@link GpsEventListener}
	 * @param gel objeto a añadir
	 */
	public void addIMUEventListener( IMUEventListener iel ) {
		listeners.add( iel );
	}
	
	/** avisa a todos los listeners con un evento */
	private void avisaListeners() {
	    for ( int j = 0; j < listeners.size(); j++ ) {
	        IMUEventListener iel = listeners.get(j);
	        if ( iel != null ) {
	          IMUEvent me = new IMUEvent(this,angulo);
	          iel.handleIMUEvent(me);
	        }
	    }
	}

	

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		ConexionSerialIMU csimu=new ConexionSerialIMU();
		csimu.ConectaPuerto("/dev/ttyUSB0");
		
		try{ Thread.sleep(15000); } catch (InterruptedException e) {};
		
		csimu.fijaFrecuencia(5);
		
		try{ Thread.sleep(15000); } catch (InterruptedException e) {};		
		csimu.fijaFrecuencia(2);
		
		try{ Thread.sleep(15000); } catch (InterruptedException e) {};
		csimu.fijaFrecuencia(0.5);
		
//		try {
//			while(true) {
//				Thread.sleep(2000);
//				byte[] men={(byte)0xfa, (byte)0xff, 0x34, 0 , (byte)0xcd};
//				System.out.println("Enviamos mensaje "+UtilMensajesIMU.hexaString(men));
//				csimu.flujoSalida.write(men);
//				csimu.flujoSalida.flush();
//			}
//		} catch (InterruptedException e) {
//			System.err.println("Se interrumpió el sleep");
//		} catch (IOException e) {
//			System.err.println("Problema al enviar mensaje:"+e.getMessage());
//		}
		
		
		

	}

}
