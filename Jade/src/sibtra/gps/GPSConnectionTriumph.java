package sibtra.gps;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.TooManyListenersException;

import sibtra.imu.UtilMensajesIMU;
import sibtra.util.EligeSerial;
import gnu.io.CommPortIdentifier;
import gnu.io.NoSuchPortException;
import gnu.io.PortInUseException;
import gnu.io.SerialPort;
import gnu.io.SerialPortEvent;
import gnu.io.SerialPortEventListener;
import gnu.io.UnsupportedCommOperationException;

/** Clase para probar los mensajes estandar de la familia Triumph
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
	private boolean esBinario=false;

	/** Banderín que indica que el mensaje es de texto */
	private boolean esTexto=false;


	/**
	 * Constructor por defecto no hace nada.
	 * Para usar el puerto hay que invocar a {@link #setParameters(SerialParameters)} 
	 * y luego {@link #openConnection()}
	 */
	public GPSConnectionTriumph() {
		super();
	}

	/**
	 * Crea conexión a GPS en puerto serial indicado.
	 * Se utilizan los parámetros <code>SerialParameters(portName, 115200, 0, 0, 8, 1, 0)</code>
	 * Si se quieren especificar otros parámetros se debe utilizar
	 * el {@link #GPSConnection() constructor por defecto}.
	 * @param portName nombre puerto donde encontrar al GPS
	 * @param baudio velocidad de la comunicacion en baudios
	 */
	public GPSConnectionTriumph(String portName) throws SerialConnectionException {
		this(portName,115200);
	}
	/**
	 * Crea conexión a GPS en puerto serial indicado.
	 * Se utilizan los parámetros <code>SerialParameters(portName, baudios, 0, 0, 8, 1, 0)</code>
	 * Si se quieren especificar otros parámetros se debe utilizar
	 * el {@link #GPSConnection() constructor por defecto}.
	 * @param portName nombre puerto donde encontrar al GPS
	 * @param baudio velocidad de la comunicacion en baudios
	 */
	public GPSConnectionTriumph(String portName, int baudios) throws SerialConnectionException {
		super(portName,baudios);
	}


	

	/**
	 * Maneja los eventos seriales {@link SerialPortEvent#DATA_AVAILABLE}.
	 * Si se recibe un mensaje completo del GPS {@link #nuevaCadenaNMEA(String)}
	 */
	public synchronized void serialEvent(SerialPortEvent e) {
		if (e.getEventType() == SerialPortEvent.DATA_AVAILABLE) {
			try {
				while (is.available() != 0) {
					int val = is.read();
					//añadimos nuevo byte recibido
					if(indFin==(buff.length-1)) {
						System.err.println("Buffer se llenó. Resetamos");
						indIni=0; indFin=-1; esBinario=false; esTexto=false;
					}
					indFin++;
					buff[indFin]= (byte) val;
					if(esTexto) {
						if ( buff[indFin] == 10 || buff[indFin]==13)
						{
							//mensaje de texto completo
							indFin--; //quitamos caracter del salto
							String menTexto=new String(buff,indIni,(indFin-indIni+1));
							if(menTexto.charAt(0)=='$')
								nuevaCadenaNMEA(menTexto);
							else
								nuevaCadenaTexto(menTexto);
							indIni=0; indFin=-1; esBinario=false; esTexto=false;
						} 
					} else if (esBinario) {
						//terminamos si ya está el tamaño
						if ( (indFin-indIni+1)==(largoMen+5) ) {
							//tenemos el mensaje estandar completo
							nuevaCadenaBinaria();						
							indIni=0; indFin=-1; esBinario=false; esTexto=false;
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
								esBinario=true;
								largoMen=largo();
							}

						}
					}
				}
			} catch (IOException ioe) {
				System.err.println("\nError al recibir los datos");
			} catch (Exception ex) {
				System.err.println("\nGPSConnection Error al procesar >"+buff+"< : " + ex.getMessage());
				ex.printStackTrace();
				indIni=-1;
			}
		}
	}

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

	/** Se invoca cuando se recibe una cadena de texto que no es NMEA */
	void nuevaCadenaTexto(String mensaje) {
		//TODO considerar mensajes de texto propietarios GREIS
	}
	
	/** Se invoca cuando se recibe una cadena binaria propietaria GREIS */
	void nuevaCadenaBinaria() {
		int larMen=indFin-indIni+1;
		//iterpretamos los mensajes
		/* RT [~~]=126
		 * struct RcvTime {5} {
  			u4 tod; // Tr modulo 1 day (86400000 ms) [ms]
  			u1 cs; // Checksum
			};
		 */
		if(buff[indIni]==(byte)'~' || buff[indIni+1]==(byte)'~') {
			if(larMen!=(5+5)) {
				System.err.println("El mensaje RT no tienen el tamaño correcto "+larMen+" Ignoramos mensaje");
				return;				
			}
			//comprobamos checksum
			byte csc=checksum8(buff, indIni, larMen-1);
			if(csc!=buff[indFin]) {
				System.err.println("Error checksum "+csc+"!="+buff[indFin]+" Ignoramos mensaje");
				return;
			}
			//el checksum es correcto
			ByteBuffer bb = ByteBuffer.allocate(4);
			bb.order(ByteOrder.LITTLE_ENDIAN);
			bb.put(buff, indIni+5, 4);
			bb.rewind();
			int tod=bb.getInt();
			System.out.println("\n\n\nMensaje RT: tod="+tod);
			return;
		}
		/* [::](ET) Epoch Time5 
   			struct EpochTime {5} {
     			u4 tod; // Tr modulo 1 day (86400000 ms) [ms]
     			u1 cs; // Checksum
   			};
		 */
		if(buff[indIni]==(byte)':' || buff[indIni+1]==(byte)':') {
			if(larMen!=(5+5)) {
				System.err.println("El mensaje ET no tienen el tamaño correcto "+larMen+" Ignoramos mensaje");
				return;				
			}
			//comprobamos checksum
			byte csc=checksum8(buff, indIni, larMen-1);
			if(csc!=buff[indFin]) {
				System.err.println("Error checksum "+csc+"!="+buff[indFin]+" Ignoramos mensaje");
				return;
			}
			//el checksum es correcto
			ByteBuffer bb = ByteBuffer.allocate(4);
			bb.put(buff, indIni+5, 4);
			bb.order(ByteOrder.LITTLE_ENDIAN);
			int tod=bb.getInt(0);
			System.out.println("Mensaje ET: tod="+tod);
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
		if(buff[indIni]==(byte)'P' || buff[indIni+1]==(byte)'O') {
			if(larMen!=(5+30)) {
				System.err.println("El mensaje PO no tienen el tamaño correcto "+larMen+" Ignoramos mensaje");
				return;				
			}
			//comprobamos checksum
			byte csc=checksum8(buff, indIni, larMen-1);
			if(csc!=buff[indFin]) {
				System.err.println("Error checksum "+csc+"!="+buff[indFin]+" Ignoramos mensaje");
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
			
			System.out.println("Mensaje PO: ("+x+","+y+","+z+") sigma="+sigma+" solType="+solType);
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
		if(buff[indIni]==(byte)'B' || buff[indIni+1]==(byte)'L') {
			if(larMen!=(5+34)) {
				System.err.println("El mensaje BL no tienen el tamaño correcto "+larMen+" Ignoramos mensaje");
				return;				
			}
			//comprobamos checksum
			byte csc=checksum8(buff, indIni, larMen-1);
			if(csc!=buff[indFin]) {
				System.err.println("Error checksum "+csc+"!="+buff[indFin]+" Ignoramos mensaje");
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

			System.out.println("Mensaje BL: ("+x+","+y+","+z+") sigma="+sigma+" solType="+solType
					+" time="+time);
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
		System.out.println("< >"+UtilMensajesIMU.hexaString(buff, indIni+5, larMen-5)+"<");
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
 */
	public static byte checksum8(byte[] buff, int ini, int largo) {
		int res=0;
		for(int i=ini; i<ini+largo; i++)
			res= (((res<<2)|(res>>>6)) ^ buff[i]) & 0xff; //nos aseguramos parte alta de  int a 0
		return (byte)((res<<2)|(res>>>6));
	}
	
	/**
	 * @param comando comando de texto a enviar al GPS
	 */
	public void comandoGPS(String comando) {
		try {
		os.write(comando.getBytes());
		System.out.println("Enviado Comando:>"+comando+"<");
		} catch (Exception e) {
			System.err.println("Problema al enviar comando Triumph:"+e.getMessage());
		}
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {

		GPSConnectionTriumph gpsC;
		
		try {
			gpsC=new GPSConnectionTriumph("/dev/ttyUSB0",9600);
			gpsC.comandoGPS("out,,jps/MF\n");

			try { Thread.sleep(5000); } catch (Exception e) {}

			gpsC.comandoGPS("em,,{jps/RT,nmea/GGA,jps/PO,jps/BL,nmea/GST,jps/ET}:10\n");
			try { Thread.sleep(100000); } catch (Exception e) {}

			gpsC.comandoGPS("dm\n");
		} catch (Exception e) {
		}
		
		System.exit(0);
	}
}