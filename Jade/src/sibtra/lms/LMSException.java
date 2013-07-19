/**
 * 
 */
package sibtra.lms;

/**
 * Para gestionar situaciones excepcionales de la comunicación con el LMS
 * @author alberto
 *
 */
public class LMSException extends Exception {
	
	public LMSException(String mensaje) {
		super(mensaje);
	}

	/**
	 * @param arg0
	 * @param arg1
	 */
	public LMSException(String arg0, Throwable arg1) {
		super(arg0, arg1);
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param arg0
	 */
	public LMSException(Throwable arg0) {
		super(arg0);
		// TODO Auto-generated constructor stub
	}

}
