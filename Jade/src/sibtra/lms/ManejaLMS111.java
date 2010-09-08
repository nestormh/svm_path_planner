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
 * mensajes de texto. La respuesta es también texto con los números representados en hexamecimal
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
	
	/** Pasea la cadena correspondiente a entero del mensaje segun definición.
	 * Los números decimales empiezan por + ó -, los hexadecimales son siempre positivos
	 * @return el entero correspondiente o {@link Integer.MIN_VALUE} si hay error 
	 */
	int str2int(String st) {
		st.trim();
		try {
		if(st.startsWith("+") || st.startsWith("-")) 
			return Integer.parseInt(st);
		else
			if((st.length()==8) && (st.charAt(0)=='F'))
				//TODO caso especial no soportado por parseInt
				return (int)((long)0xffffffff+Long.parseLong(st,16)+1);
			else
				return Integer.parseInt(st, 16);
		} catch (NumberFormatException ne) {
			System.err.println("Error al convertir a entero "+st+"\n"+ne.getLocalizedMessage());
			ne.printStackTrace();
		}
		return Integer.MIN_VALUE;
	}
	
	/** Parsea los mensajes <code>sRA LMDscandata</code> y <code>sSN LMDscandata</code>*/
	public boolean parseaBarrido(String resp) {
		if(!resp.startsWith("sRA LMDscandata")
				&& !resp.startsWith("sSN LMDscandata")) return false; //no es barrido
		String[] campos=resp.split("\\s");
		String log="";
		int ca=2;
		log+="\n"+ca+": "+"Version="+str2int(campos[ca++]) + " Dev num="+str2int(campos[ca++])
			+ " Ser num="+str2int(campos[ca++])+" Stat="+str2int(campos[ca++]);
		log+="\n"+ca+": "+"Men count="+str2int(campos[ca++]) + " Scan Count="+str2int(campos[ca++]);
		log+="\n"+ca+": "+"PwUpDur="+str2int(campos[ca++]) + " TrDur="+str2int(campos[ca++]);
		ca+=2;
		//TODO INput stat y oput stat ocupan 2 campos cada uno ??
		log+=" InStat="+str2int(campos[ca++])+" OutStat="+str2int(campos[ca++]);
		ca+=2; //reservado
		log+="\n"+ca+": "+"Scan Fre="+str2int(campos[ca++]) + " Measu Fre="+str2int(campos[ca++]);
		
		int numEnc=str2int(campos[ca++]);
		log+="\n"+(ca-1)+": "+"NumEnc="+numEnc;
		for(int i=0; i<numEnc;i++)
			log+=" Enc"+1+": Ticks="+str2int(campos[ca++])+" ticks/mm="+str2int(campos[ca++]);
		
		int numChanels16=str2int(campos[ca++]);
		log+="\n"+(ca-1)+": "+"NumChanels16="+numChanels16;
		for(int i=0; i<numChanels16; i++) {
			log+="\n"+ca+": Chan"+i+" DataCont="+campos[ca++]+" ScanFact="+str2int(campos[ca++])
			+" ScanOffset="+str2int(campos[ca++])+" StarAng="+str2int(campos[ca++])+" AngStep="+str2int(campos[ca++]);
			int numData=str2int(campos[ca++]);
			log+="\n\t"+(ca-1)+": "+"NumData="+numData;
			ca+=numData;
		}
		
		int numChanels8=str2int(campos[ca++]);
		log+="\n"+(ca-1)+": "+"NumChanels8="+numChanels8;
		for(int i=0; i<numChanels8; i++) {
			log+="\n"+ca+": Chan"+i+" DataCont="+campos[ca++]+" ScanFact="+str2int(campos[ca++])
			+" ScanOffset="+str2int(campos[ca++])+" StarAng="+str2int(campos[ca++])+" AngStep="+str2int(campos[ca++]);
			int numData=str2int(campos[ca++]);
			log+="\n\t"+(ca-1)+": "+"NumData="+ (numData+0);
			ca+=numData;
		}

		int position=str2int(campos[ca++]);
		log+="\n"+(ca-1)+": "+"position="+position;
		if(position!=0) {
			ca+=7;
		}
		
		int name=str2int(campos[ca++]);
		log+="\n"+(ca-1)+": "+"name="+name;
		if(name!=0) {
			log+=" len="+str2int(campos[ca++])+" Name="+campos[ca++];
		}
		
		int coment=str2int(campos[ca++]);
		log+="\n"+(ca-1)+": "+"coment="+coment;
		if(coment!=0) {
			log+=" len="+str2int(campos[ca++])+" coment="+campos[ca++];
		}
		
		int timeInfo=str2int(campos[ca++]);
		log+="\n"+(ca-1)+": "+"timeInfo="+timeInfo;
		if(timeInfo!=0) {
			log+=" Year="+str2int(campos[ca++])+" Month="+str2int(campos[ca++])+" Day="+str2int(campos[ca++])
			+" Hour="+str2int(campos[ca++])+" Min="+str2int(campos[ca++])+" Sec="+str2int(campos[ca++])+" uSec="+str2int(campos[ca++]);
		}
		
		int eventInfo=str2int(campos[ca++]);
		log+="\n"+(ca-1)+": "+"eventInfo="+eventInfo;
		if(eventInfo!=0) {
			ca+=4;
		}
		
		if(ca!=campos.length) {
			System.err.println("Numero incorrecto de campos");
		}
		
		System.out.println(log);
		return true;
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
	
	public String enviaEspera(String men) {
		System.out.println("\nEnviamos mensaje:>"+men+"<");
		enviaMensaje(men);
		System.out.println("Esperamos respuesta");
		String rec=leeMensaje();
		System.out.print("  >"+rec+"<\n  >");
		String[] campos=rec.split("\\s");
		for(String ca:campos) {
			try { 
				System.out.print(str2int(ca.trim())+" "); 
			} catch (Exception e) {
				System.out.print(ca+" ");
			}
		}
		System.out.println("<");
		return rec;
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {

		
		ManejaLMS111 m111=new ManejaLMS111();
		m111.parseaBarrido("sRA LMDscandata 1 1 8D4C04 0 0 45FF 37 280F6 28A73 0 0 7 0 0 1388 168 0 1 DIST1 3F800000 00000000 FFF92230 1388 21D 7E 82 7D 79 83 76 7A 84 86 79 79 82 87 82 8A 90 7B 86 9B 88 7E 8C 97 A0 9F 9D 9D 9E AF A7 B2 A2 AE BA BC CA CA D2 E1 CD DF D1 E5 D3 E4 F4 FC 104 10C 108 11D 128 12B 147 143 15A 168 175 17B 184 18E 193 186 180 194 1A7 178 18B 197 19B 195 187 16B 177 179 172 170 17E 185 197 1A3 1F7 25E 27A 267 23E 1D3 1CA 1B6 1CA 1C1 1BF 1B8 1BA 1AD 191 19D 194 19F 192 17A 198 1BC 1F0 1E1 1E2 1CE 1CC 1BF 1C4 1C1 1FD 245 268 264 27B 276 283 273 283 29B 304 557 5AC 5BA 5D1 5CF 5DB 5EF 607 612 60D 612 620 629 64D 65D 659 66C 698 6A0 6AA 6C0 6D0 6CE 6DB 6F4 70C 70D 733 742 75A 76F 775 792 7A4 7B3 7E3 7E6 81E 82A 841 858 87C 8A8 8C0 8DD 907 941 953 943 923 923 8F6 8EB 8D3 8B9 8A1 8A8 86D 85C 84A 843 83A 81A 814 7F6 7F7 7D1 7D0 7C8 7AB 79D 799 792 777 770 767 752 74D 749 72F 72D 716 716 6FF 705 6F4 6FC 6DA 6E6 6CF 6CA 6B7 6AF 6AA 6A7 6AA 6A3 68E 687 6A8 67B 68B 66B 665 663 662 653 65B 64B 646 658 643 63A 634 62C 622 62B 628 62D 620 61A 62F 61A 617 612 608 60A 60C 60C 616 605 600 60B 61A 601 5FE 5F6 5FA 600 5FB 604 5FA 5F9 5F2 60B 5F2 5F5 5F0 5F5 5F0 5F8 5FB 5F1 5FC 5FC 5F3 605 5F8 5FF 5F9 5F9 600 5FB 610 5FE 60C 616 60B 609 610 610 60E 615 622 624 61F 61F 62B 632 628 62E 635 631 63F 642 64F 651 651 653 666 65E 65F 668 66F 677 68A 692 68C 688 698 6AA 6B5 6A7 6AE 6B4 6C0 6CB 6E5 6D9 6EB 6FB 6F5 701 704 711 71D 743 73A 753 774 78D 78B 79C 7A6 7A2 7B8 7D8 7D6 7E3 7FB 811 81C 826 841 82E 840 851 86C 86B 87A 88C 8A8 8B7 8CF 8E8 8F6 910 925 940 963 97D 995 9AF 9D4 9EF A01 A33 A3D A69 A8F AB6 AC7 AF9 B28 B4B B81 B94 BC8 BEC C23 C50 C7B C7E C9A CC8 D6B D51 D4B D38 D16 CF8 CF2 CE8 CD3 CC2 CAF CAF CA7 CB4 C9C C91 C7D C64 C69 C3F C3D C3C C38 C20 C11 C0F C06 BE4 BCB AE3 8D7 75F 768 76E 777 767 767 770 76E 768 761 764 757 75B 765 777 77B 773 765 8A1 9D9 B24 4C4 462 448 43D 45C 457 44E 41E 421 446 437 424 429 440 4D8 4A0 45E 427 454 450 460 467 472 462 46E 478 487 47A 48F 481 490 489 488 462 3E6 1B6 15A 13B F9 ED 102 EC D8 E2 DF E1 C8 D5 CF CD C8 C7 C3 BC BD A6 A2 9D 9B A6 A9 93 A9 9B 9A 9A 96 89 95 7A 90 8A 93 81 85 71 89 85 7E 75 72 64 6E 7A 74 75 71 7C 76 67 6D 67 0 0 1 7 Derecha 0 1 7B2 1 1 0 2A 27 28870 0");
		
		System.exit(0);
		System.out.println("Tratamos de conectar");
		m111.conecta("192.168.0.3", 2111);

		String men;
		
		m111.enviaEspera("sMN SetAccessMode 03 F4724744");

		//configuracion escaneo
		m111.enviaEspera("sRN LMPscancfg");

		//configuracion mensaje de datos escaneo
//		m111.enviaEspera("sWN LMDscandatacfg 01 00 0 1 00 00 00 00 00 +1");
		m111.enviaEspera("sWN LMDscandatacfg 01 00 0 1 0 00 00 1 1 1 1 +1");


		//7-segmento encendido
		m111.enviaEspera("sMN mLMLSetDisp 07");

		//Pedimos que comienze a medir
		m111.enviaEspera("sMN LMCstartmeas");
		
		String rec;
		do {
			//Miramos el status
			try { Thread.sleep(1000); } catch (Exception e) {}
			rec=m111.enviaEspera("sRN STlms");
		} while(rec.charAt(10)!='7');
		
		//Pedimos una medida
		m111.enviaEspera("sRN LMDscandata");

		//Pedimos una medida
		m111.enviaEspera("sRN LMDscandata");

		//Pedimos una medida
		m111.enviaEspera("sRN LMDscandata");

		//paramos el LMS
//		m111.enviaEspera("sMN LMCstopmeas");
		
		m111.desconecta();
		
	}

}
