/**
 * 
 */
package sibtra.lms;


/**
 * @author alberto
 *
 */
public class ManejaTelegramasJNI extends ManejaTelegramas {

	/**
	 * Para abrir y configurar puerto serie
	 * 
	 * @param NombrePuerto nombre del puerto serie a utilizar
	 */
	public native boolean ConectaPuerto(String NombrePuerto); 
	
	/** @return si el pueto ha sido correctamente inicializado por {@link #ConectaPuerto(String)} */
	public native boolean isInicializado();

	/** 
	 * Trata de fijar la velocidad de transmisisón del puerto al indicado
	 * @param baudrate velocidad desead
	 * @return ture si la velocidad es valida y se consiguió el cambio.
	 */
	public native boolean setBaudrate(int baudrate);
	

	/**
	 * Espera la llegada de un telegrama por la serial y extrae el mensaje
	 * @param milisTOut especifica milisegundos para dar <i>time out</i>. 
	 *   0 significa no time out.
	 * @return mensaje contenido en telegrama
	 */
	public native byte[] LeeMensaje(int milisTOut);

	/**
	 * Construye a telegarama a partir del mensaje pasado y lo envía.
	 * NO espera la confirmación de que ha sido bien recibido.
 	 * @param mensaje a enviar
 	 * @return true si se evió correctamente 
	 */
	public native boolean EnviaMensajeSinConfirmacion(byte[] mensaje);
	
	/** Espera la confirmación
	 * @param milisTOut especifica milisegundos para dar <i>time out</i>.
	 * @return true si se recibe confirmación
	 */
	public native boolean esperaConfirmacion(int milisTOut);

	/** Vacia todo el buffer de entrada de la serial */
	public native void purgaBufferEntrada();
	
	/**
	 * Cierra el puerto
	 */
	public native boolean cierraPuerto();

	static {
		System.out.println("java.library.path:"
				+System.getProperty("java.library.path"));
       //System.loadLibrary( "sibtra_ManejaTelegramasJNI" );
		String libreria=System.getProperty("user.dir")+"/lib/lms/sibtra_lms_ManejaTelegramasJNI.so";
		System.out.println("Tratamos de cargar: "+libreria);
	   System.load(libreria);
    }
	
	
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		String		      defaultPort = "/dev/ttyS1";
		ManejaTelegramasJNI	MT = null;
		
		if (args.length > 0) {
			defaultPort = args[0];
		} 

		MT = new ManejaTelegramasJNI();
		MT.ConectaPuerto(defaultPort);

		byte[] MenBarrido={0x30, 0x01}; 
		//byte[] MenBarrido={0x37, 0x01, 0x00, (byte)0xc0, 0x00}; 
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
