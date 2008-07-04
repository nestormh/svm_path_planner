/**
 * 
 */
package sibtra.lms;


/**
 * @author alberto
 *
 */
public class ManejaTelegramasJNI implements ManejaTelegramas {

	/**
	 * Para abrir y configurar puerto serie
	 * 
	 * @param NombrePuerto nombre del puerto serie a utilizar
	 */
	public native boolean ConectaPuerto(String NombrePuerto); 
	
	/**
	 * Espera la llegada de un telegrama por la serial y extrae el mensaje
	 * @return mensaje contenido en telegrama
	 */
	public native byte[] LeeMensaje();

	/**
	 * Construye a telegarama a partir del mensaje pasado y lo envía.
	 * Espera la confirmación de que ha sido bien recibido.
	 * 
	 * @return true si se recibe confirmación
	 */
	public native boolean EnviaMensaje(byte[] mensaje);
	
	/**
	 * Cierra el puerto
	 */
	public native boolean cierraPuerto();
    
	static {
		System.out.println("java.library.path:"
				+System.getProperty("java.library.path"));
       //System.loadLibrary( "sibtra_ManejaTelegramasJNI" );
		String libreria=System.getProperty("user.dir")+"/../lib/sibtra_lms_ManejaTelegramasJNI.so";
		System.out.println("Tratamos de cargar: "+libreria);
	   System.load(libreria);
    }
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		String		      defaultPort = "/dev/ttyS0";
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
