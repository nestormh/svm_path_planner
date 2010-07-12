/**
 * 
 */
package sibtra.lms;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.net.Socket;

/**
 * Clase para el acceso a los LMS111. Usa una conexión TCP/IP al puerto del dispositivo y a partir de ahí se envían
 * mensajes de texto. La respueste es también texto con los números representados en hexamecimal
 * Las direcciónes IP están en el rango 192.168.0.X, por lo que es necesario cofigurar interfáz de la máquina en ese rango
 * Para Portatil 9 basta con hacer 
 * <code>ifconfig eth0 add 192.168.0.10 netmask 255.255.255.0</code>
 * 
 * @author alberto
 *
 */
public class ManejaLMS111 {
	
	public static final char STX=2;
	public static final char ETX=3;
	/** Tamaño máximo del mensaje */
	private static final int MAXMENSAJE = 5000;

	private BufferedReader in;
	private PrintStream out;
	/** Buffer en la que se van almacenando los trozos de mensajes que se van recibiendo */
	private int buff[] = new int[MAXMENSAJE];
	
	private boolean conectado=false;
	private Socket sock;

	
	
	public boolean conecta(String host, int port) {
		try {
			sock=new Socket(host,port);
			in=new BufferedReader(new InputStreamReader(sock.getInputStream()));
			out=new PrintStream(sock.getOutputStream());
			conectado=true;
			return true;
		} catch (IOException e) {
			System.err.println("Problemas al conectara con "+host+" en puerto:"+port+" :"
					+e.getLocalizedMessage());
			e.printStackTrace();
		}
		return false;
	}

	public void desconecta() {
		if(!conectado)
			return;
		try {
			in.close();
			out.close();
			sock.close();
			conectado=false;
		} catch (IOException e) {
			System.err.println("Problemas al cerrar la conexion:"+e.getLocalizedMessage());
			e.printStackTrace();
		}
	}
	public void enviaMensaje(String msg){
		if(!conectado)
			throw new IllegalStateException("NO estamos conectados");
		out.println(STX+msg+ETX);
	}
	
	/**
	 * Recibe un mensaje del LSM
	 * @return mensaje sin STX ni ETX
	 */
	public String leeMensaje() {
		if(!conectado)
			throw new IllegalStateException("NO estamos conectados");
		int ca;
		int indCa=0; //índice del caracter actual
		int despreciados=0; //número de caracteres despreciados
		try {
			while((ca=in.read()) != -1) {
				if(indCa<1 && ca!=STX) {
					//despreciamos el caracter
					despreciados++;
					continue;
				}
				buff[indCa++]=ca;
				if(ca==ETX)
					break; //tenemos el fin de texto
			}
			if(despreciados>0)
				System.err.println("Se despreciaron: "+despreciados);
			if(ca==-1) {
				System.err.println("Encontrado Fin de archivo");
			}
			return new String(buff,1,indCa-2);
		} catch (IOException e) {
			System.err.println("Error de I/O recibiendo mensaje:"+e.getLocalizedMessage());
			e.printStackTrace();
		} catch (IndexOutOfBoundsException e) {
			System.err.println("Error al convertir a string");
		}
		return null;
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {

		ManejaLMS111 m111=new ManejaLMS111();
		System.out.println("Tratamos de conectar");
		m111.conecta("192.168.0.3", 2111);

		String men;
		
		men="sMN SetAccessMode 03 F4724744";
		System.out.println("Enviamos mensaje:>"+men+"<");
		m111.enviaMensaje(men);
		System.out.println("Esperamos respuesta");
		System.out.println("Recibido mensaje:>"+m111.leeMensaje()+"<");

		//configuracion escaneo
		men="sRN LMPscancfg";
		System.out.println("Enviamos mensaje:>"+men+"<");
		m111.enviaMensaje(men);
		System.out.println("Esperamos respuesta");
		System.out.println("Recibido mensaje:>"+m111.leeMensaje()+"<");

		//7-segmento encendido
		men="sMN mLMLSetDisp 07";
		System.out.println("Enviamos mensaje:>"+men+"<");
		m111.enviaMensaje(men);
		System.out.println("Esperamos respuesta");
		System.out.println("Recibido mensaje:>"+m111.leeMensaje()+"<");

		//Pedimos que comienze a medir
		men="sMN LMCstartmeas";
		System.out.println("Enviamos mensaje:>"+men+"<");
		m111.enviaMensaje(men);
		System.out.println("Esperamos respuesta");
		System.out.println("Recibido mensaje:>"+m111.leeMensaje()+"<");
		
		String rec;
		do {
			//Miramos el status
			try { Thread.sleep(1000); } catch (Exception e) {}
			men="sRN STlms";
			System.out.println("Enviamos mensaje:>"+men+"<");
			m111.enviaMensaje(men);
			System.out.println("Esperamos respuesta");
			rec=m111.leeMensaje();
			System.out.println("Recibido mensaje:>"+rec+"<");
		} while(rec.charAt(10)!='7');
		
		//Pedimos una medida
		men="sRN LMDscandata";
		System.out.println("Enviamos mensaje:>"+men+"<");
		m111.enviaMensaje(men);
		System.out.println("Esperamos respuesta");
		System.out.println("Recibido mensaje:>"+m111.leeMensaje()+"<");

		//paramos el LMS
		men="sMN LMCstopmeas";
		System.out.println("Enviamos mensaje:>"+men+"<");
		m111.enviaMensaje(men);
		System.out.println("Esperamos respuesta");
		System.out.println("Recibido mensaje:>"+m111.leeMensaje()+"<");
		
		m111.desconecta();
		
	}

}
