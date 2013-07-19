/**
 * 
 */
package sibtra.lms;

import java.io.*;
import java.util.*;

import gnu.io.*;

/**
 * Clase que realiza la comunicación con el LMS a través de la serial
 * @author alberto
 *
 */
public class ManejaTelegramasIO extends ManejaTelegramas {
	private static final int MaxLen = 812;
	private CommPortIdentifier idPuertoCom;
	private SerialPort puertoSerie;
	private InputStream flujoEntrada;
	private OutputStream flujoSalida;
	private byte[] buf;
	/**
	 * Para saber si ya se ha inicializado el puerto
	 */ 
	private boolean inicializado=false;

	/* (non-Javadoc)
	 * @see sibtra.lms.ManejaTelegramas#ConectaPuerto(java.lang.String)
	 */
	public boolean ConectaPuerto(String NombrePuerto) {
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
			puertoSerie.setSerialPortParams(38400, SerialPort.DATABITS_8, 
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

		inicializado=true;
		buf=new byte[MaxLen]; // creamos del tamaño máximo de telegrama
		return true;
	}

	/** @return si el pueto ha sido correctamente inicializado por {@link #ConectaPuerto(String)} */
	public boolean isInicializado() {
		return inicializado;
	}

	/** 
	 * Trata de fijar la velocidad de transmisisón del puerto al indicado
	 * @param baudrate velocidad desead
	 * @return ture si la velocidad es valida y se consiguió el cambio.
	 */
	public boolean setBaudrate(int baudrate) {
		//Para cambiar velocidad tenemos que fijar todos los parámetros
		//Si la velocidad no es válida dará una exepción
		try {
			puertoSerie.setSerialPortParams(baudrate, SerialPort.DATABITS_8, 
					SerialPort.STOPBITS_1, 
					SerialPort.PARITY_NONE);
		} catch (UnsupportedCommOperationException e) {
			System.err.println("\n No puedo fijar la nueva velocidad "+baudrate+" la puerto");
			return false;			
		}
		return true;
	}

	/* (non-Javadoc)
	 * @see sibtra.lms.ManejaTelegramas#LeeMensaje()
	 */
	public byte[] LeeMensaje(int milisTOut) {
		//TODO ver como se puede fijar TimeOut
		if(!inicializado)
			return null;  //lo suyo sería una excepción :-(
		int res; //cuantos bytes hemos recibido
		try {
			res=0;
			if((res=flujoEntrada.read(buf, 0, 1))==0) {
				System.err.println("Timeout esperando comienzo telegrama");
				return null;			
			}
			if(buf[0]!=2){
				System.err.println("No es comienzo de telegrama:"+buf[0]+" res: "+res);
				return null;			
			}
			System.err.printf("\n STX: %02X; disponibles %d",buf[0],flujoEntrada.available());
			
			//leemos hasta el tamaño y comando
			while(res<5) {
				int bl;
				if((bl=flujoEntrada.read(buf, res, 5-res))==0) {
					System.err.println("Timeout esperando tamaño");
					return null;
				}
				res+=bl;
			}         

			System.err.printf(" Addr: %02X;",buf[1]);
			
			int len=UtilMensajes.men2Word(buf,2);
			System.err.printf(" Len: (%d) ",len);
			System.err.printf(" comando: %02X; \n",buf[4]);

			if((len+6)>MaxLen) {
				System.err.println("Longitud supera el tamaño máxmo !!! ");
				return null;
			}
			//Conseguimos el resto del mensaje
			while(res<(len+6)) {
				int bl;
				int faltan=(len+6)-res;
				if((bl=flujoEntrada.read(buf,res,(faltan>200)?200:faltan))==0){
					System.err.println("Timeout esperando resto del mensaje: "+res+" de "+(len+6));
					//return null;
				}
				res+=bl;
				System.err.println("Recibidos "+bl+" del resto, tenemos "+res+" nos faltan "+((len+6)-res));
			}         

			System.err.printf("Mensaje completo\n");

//			//Volcamos en hexadecimal
//			for (int i=0; i<res; i++)
//				System.err.printf(" %02X",buf[i]);
//			System.err.printf("\n");
//
//			System.err.println("\nMensaje completo: "+UtilMensajes.hexaString(buf,0,res));
//			System.err.println("\nMensaje completo con utils: "+java.util.Arrays.toString(buf));

			
//			//Volcamos caracter
//			for (int i=0; i<res; i++)
//				if(buf[i]>=' ' && buf[i]<=127)
//					System.err.printf("%c",buf[i]);
//				else
//					System.err.printf(" %02X",buf[i]);

			System.err.printf("Respuesta: %02X\n",buf[4]);
//			System.err.printf("\n Respuesta: %02X;\nDatos:\n",buf[4]);
//			//Volcamos caracter
//			for (int i=5; i<(res-3); i++)
//				if(buf[i]>=' ' && buf[i]<=127)
//					System.err.printf("%c",buf[i]);
//				else
//					System.err.printf(" %02X",buf[i]);
//			System.err.printf("\n");

//			//Volcamos en hexadecimal
//			for (int i=5; i<(res-3); i++)
//				System.err.printf(" %02X",buf[i]);
//			System.err.println("\nDato en hexa: "+UtilMensajes.hexaString(buf,5,res-3-5+1));
//			System.err.printf("\n Status: %02X;",buf[res-3]);
//
//			System.err.printf("\nCHKS: %02X%02X;",buf[res-1],buf[res-2]);
			
			int CRCcal=UtilMensajes.CalculaCRC(buf,res-2);
			int CRCtele=UtilMensajes.men2Word(buf, res-2);
			
			System.err.printf("CRC y Calculado: %04X, %04X\n",CRCcal,CRCtele);
			
			if(CRCcal!=CRCtele) {
				System.err.println("El CRC NO es correcto");
				return null;
			}
//			else {
//				System.err.println("El CRC  ES correcto");
//			}

			byte[] mensaje=new byte[len];
			//copiamos del buffer al array que se va a devolver
			System.arraycopy(buf, 4, mensaje, 0, len);
			return mensaje;
		} catch (IOException e) {
			System.err.println("Problema al acceder al leer del puerto: "+e.getMessage());
			return null;
		}
	}

	
	public boolean EnviaMensajeSinConfirmacion(byte[] mensaje) {

		if(!inicializado)
			return false; //lo suyo sería una excepción :-(

		int TamMen=mensaje.length;
		if (TamMen<=0){
			System.err.println("Mensaje debe tener al menos 1 byte");
			return false;
		}
		//copiamos mensaje a bufer
		System.arraycopy(mensaje, 0, buf, 4, TamMen);

		//Completamos telegrama
		buf[0]=2;
		buf[1]=0;
		buf[2]=(byte)(TamMen%256);
		buf[3]=(byte)(TamMen/256);

		int CRC=UtilMensajes.CalculaCRC(buf,TamMen+4);
		buf[4+TamMen]=(byte)(CRC%256);
		buf[4+TamMen+1]=(byte)(CRC/256);

		System.err.printf("\nTelegrama a enviar:");
		//Volcamos en hexadecimal
		for (int i=0; i<TamMen+6; i++)
			System.err.printf(" %02X",buf[i]);
		System.err.printf("\nProcedemos a enviarlo:");
		
		//Purgamos entrada
		try {
			int dispo=flujoEntrada.available();
			if(dispo>0){
				System.err.println("Se purgan "+dispo+" bytes");
				flujoEntrada.skip(dispo);
			}
		} catch (IOException e1) {
			System.err.println("No se pudo purgar la entrada");
			e1.printStackTrace();
		}

		try {
			flujoSalida.write(buf);
			flujoSalida.flush();
		} catch (IOException e) {
			System.err.println("No se envió el mensaje");
			return false;
		}	
//		System.err.printf("\nEnviado Correctamente\n");
		return true;
	}
	
	public boolean esperaConfirmacion(int milisTOut) {
		//TODO ver como se puede fijar TimeOut
		//Esperamos confirmación
		try {
			if(flujoEntrada.read(buf, 0, 1)==0) {
				System.err.println("Timeout esperando confirmación.");
				return false;
			}
			if(buf[0]==0x06){
//				System.err.println("Telegrama confirmado");
				return true;			
			}
			else if(buf[0]==0x15) {
				System.err.println("Telegrama NO CONFIRMADO");
				return false;
			} else {
				System.err.println("No se recibió confirmación ni desconfirmación ??"+buf[0]);
				return false;			
			}
		} catch (IOException e) {
			System.err.println("Error esperando la confirmación");
			return false;
		}
	}

	/** Vacia todo el buffer de entrada de la serial */
	public void purgaBufferEntrada() {
		//TODO Pendiente implementación
	}

	
	public boolean cierraPuerto() {
		if(!inicializado)
			return false;
		puertoSerie.close();
		return true;
	}
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		boolean		      portFound = false;
		String		      defaultPort = "/dev/ttyS0";
		CommPortIdentifier portId;
		ManejaTelegramasIO	MT = null;
		
		if (args.length > 0) {
			defaultPort = args[0];
		} 

		Enumeration portList = CommPortIdentifier.getPortIdentifiers();

		while (portList.hasMoreElements()) {
			portId = (CommPortIdentifier) portList.nextElement();
			if (portId.getPortType() == CommPortIdentifier.PORT_SERIAL) {
				if (portId.getName().equals(defaultPort)) {
					System.err.println("Found port: "+defaultPort);
					portFound = true;
				} 
			} 
		} 
		if (!portFound) {
			System.out.println("port " + defaultPort + " not found.");
			return;
		} 

		MT = new ManejaTelegramasIO();
		MT.ConectaPuerto(defaultPort);
		
		//byte[] MenBarrido={0x30, 0x01}; 
		byte[] MenBarrido={0x37, 0x01, 0x00, (byte)0xc0, 0x00}; 
		
		int numbar=1;
		while(numbar<=5) {
			System.err.println("Intentamos con barrido "+numbar);
			MT.EnviaMensaje(MenBarrido);
			if(MT.LeeMensaje()==null)
				System.err.println("No se recibió bien el mensaje.");

			//esperamos un rato
			try {
				Thread.sleep(1000);
			} catch (Exception e) {
				// no hacemos nada si se interrumpe
			}
			numbar++;
		}

		System.err.println("TERMINAMOS");
		MT.cierraPuerto();
		System.exit(0);
	}

}

